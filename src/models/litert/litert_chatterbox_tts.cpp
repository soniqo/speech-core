#include "speech_core/models/litert_chatterbox_tts.h"

#include <litert/c/litert_model.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <regex>
#include <stdexcept>

namespace speech_core {

namespace {
std::vector<char> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Chatterbox: cannot open " + p);
    std::streamsize n = f.tellg(); f.seekg(0);
    std::vector<char> b((size_t)n); f.read(b.data(), n); return b;
}
template <class T> std::vector<T> read_vec(const std::string& p) {
    auto b = read_file(p); std::vector<T> v(b.size() / sizeof(T)); std::memcpy(v.data(), b.data(), b.size()); return v;
}
int parse_idx(const std::string& nm, const std::string& tag) {
    std::smatch m; std::regex re(tag + R"(_(\d+))");
    if (std::regex_search(nm, m, re)) return std::stoi(m[1]);
    throw std::runtime_error("no " + tag + "_N in " + nm);
}
}  // namespace

// A loaded graph + its signature name->logical-index maps (ai-edge scrambles order).
struct LiteRTChatterboxTts::Graph {
    LiteRtModel model = nullptr; LiteRtCompiledModel compiled = nullptr; LiteRtSignature sig = nullptr;
    std::vector<int> in_l, out_l;
    void load(const std::string& path, bool hw) {
        LiteRTEngine::get().load(path, hw, &model, &compiled);
        litert_check(LiteRtGetModelSignature(model, 0, &sig), "sig");
        LiteRtParamIndex ni = 0, no = 0;
        litert_check(LiteRtGetNumSignatureInputs(sig, &ni), "ni");
        litert_check(LiteRtGetNumSignatureOutputs(sig, &no), "no");
        in_l.resize(ni); out_l.resize(no);
        for (LiteRtParamIndex i = 0; i < ni; ++i) { const char* n; LiteRtGetSignatureInputName(sig, i, &n); in_l[i] = parse_idx(n, "args"); }
        for (LiteRtParamIndex i = 0; i < no; ++i) { const char* n; LiteRtGetSignatureOutputName(sig, i, &n); out_l[i] = parse_idx(n, "output"); }
    }
    ~Graph() { if (compiled) LiteRtDestroyCompiledModel(compiled); if (model) LiteRtDestroyModel(model); }
};

LiteRTChatterboxTts::LiteRTChatterboxTts(const std::string& dir, bool hw) {
    prefill_ = std::make_unique<Graph>(); step_ = std::make_unique<Graph>();
    enc_ = std::make_unique<Graph>(); est_ = std::make_unique<Graph>(); hift_ = std::make_unique<Graph>();
    prefill_->load(dir + "/chatterbox-t3-prefill.tflite", hw);
    step_->load(dir + "/chatterbox-t3-step.tflite", hw);
    enc_->load(dir + "/chatterbox-flow-encoder.tflite", hw);
    est_->load(dir + "/chatterbox-flow-estimator.tflite", hw);
    hift_->load(dir + "/chatterbox-hift.tflite", hw);
    tok_ = std::make_unique<ChatterboxTokenizer>(dir + "/grapheme_mtl_merged_expanded_v1.json", dir + "/Cangjie5_TC.json");
    cond_emb_ = read_vec<float>(dir + "/cond_emb.bin");
    spks_ = read_vec<float>(dir + "/spks.bin");
    prompt_token_ = read_vec<int32_t>(dir + "/prompt_token.bin");
    prompt_feat_ = read_vec<float>(dir + "/prompt_feat.bin");
    cond_len_ = (int)cond_emb_.size() / kHidden;
    prompt_feat_frames_ = (int)prompt_feat_.size() / kMel;
}
LiteRTChatterboxTts::~LiteRTChatterboxTts() = default;
void LiteRTChatterboxTts::cancel() { cancelled_.store(true); }

// ---- T3: text IDs -> speech tokens (sampled AR loop) ----
std::vector<int> LiteRTChatterboxTts::generate_speech_tokens(const std::vector<long>& text_ids, std::mt19937& rng) {
    LiteRtEnvironment env = LiteRTEngine::get().env();
    const int L = kLayers, NKV = kKv, HD = kHeadDim;
    const int prefix_len = cond_len_ + (int)text_ids.size();

    std::vector<std::vector<float>> K(L), V(L);
    for (int l = 0; l < L; ++l) { K[l].assign((size_t)NKV * kStepCache * HD, 0.f); V[l].assign((size_t)NKV * kStepCache * HD, 0.f); }

    // prefill
    {
        std::vector<int64_t> ids(kTextPad, 0);
        for (size_t i = 0; i < text_ids.size(); ++i) ids[i] = text_ids[i];
        auto t_ids = make_type(kLiteRtElementTypeInt64, {1, kTextPad});
        auto t_cond = make_type(kLiteRtElementTypeFloat32, {1, cond_len_, kHidden});
        auto t_log = make_type(kLiteRtElementTypeFloat32, {1, kPrefillT, kVocab});
        auto t_kv = make_type(kLiteRtElementTypeFloat32, {1, NKV, kPrefillT, HD});
        LiteRtHostBuffer in_ids(env, t_ids, ids.size() * 8, ids.data());
        LiteRtHostBuffer in_cond(env, t_cond, cond_emb_.size() * 4, cond_emb_.data());
        LiteRtHostBuffer out_log(env, t_log, (size_t)kPrefillT * kVocab * 4);
        std::vector<LiteRtHostBuffer> kb, vb; kb.reserve(L); vb.reserve(L);
        for (int l = 0; l < L; ++l) kb.emplace_back(env, t_kv, (size_t)NKV * kPrefillT * HD * 4);
        for (int l = 0; l < L; ++l) vb.emplace_back(env, t_kv, (size_t)NKV * kPrefillT * HD * 4);
        std::vector<LiteRtTensorBuffer> ins(prefill_->in_l.size()), outs(prefill_->out_l.size());
        for (size_t i = 0; i < ins.size(); ++i) ins[i] = prefill_->in_l[i] == 0 ? in_ids.raw() : in_cond.raw();
        for (size_t j = 0; j < outs.size(); ++j) { int m = prefill_->out_l[j]; outs[j] = m == 0 ? out_log.raw() : (m <= L ? kb[m - 1].raw() : vb[m - 1 - L].raw()); }
        litert_check(LiteRtRunCompiledModel(prefill_->compiled, 0, ins.size(), ins.data(), outs.size(), outs.data()), "prefill");
        std::vector<float> tmp((size_t)NKV * kPrefillT * HD);
        for (int l = 0; l < L; ++l) {
            kb[l].read(tmp.data(), tmp.size() * 4);
            for (int h = 0; h < NKV; ++h) std::memcpy(&K[l][(size_t)h * kStepCache * HD], &tmp[(size_t)h * kPrefillT * HD], (size_t)prefix_len * HD * 4);
            vb[l].read(tmp.data(), tmp.size() * 4);
            for (int h = 0; h < NKV; ++h) std::memcpy(&V[l][(size_t)h * kStepCache * HD], &tmp[(size_t)h * kPrefillT * HD], (size_t)prefix_len * HD * 4);
        }
    }

    // sampled AR loop
    auto t_s1 = make_type(kLiteRtElementTypeInt64, {1, 1});
    auto t_pos = make_type(kLiteRtElementTypeInt64, {1});
    auto t_log1 = make_type(kLiteRtElementTypeFloat32, {1, 1, kVocab});
    auto t_kv = make_type(kLiteRtElementTypeFloat32, {1, NKV, kStepCache, HD});
    const size_t kvb = (size_t)NKV * kStepCache * HD * 4;
    std::vector<int> gen;
    std::vector<float> logits(kVocab);
    int tok = kBos;
    for (int s = 0; s < max_tokens_ && !cancelled_.load(); ++s) {
        int64_t t64 = tok, sp = s, cp = prefix_len + s;
        LiteRtHostBuffer in_tok(env, t_s1, 8, &t64), in_sp(env, t_s1, 8, &sp), in_cp(env, t_pos, 8, &cp);
        std::vector<LiteRtHostBuffer> iK, iV, oK, oV; iK.reserve(L); iV.reserve(L); oK.reserve(L); oV.reserve(L);
        for (int l = 0; l < L; ++l) iK.emplace_back(env, t_kv, kvb, K[l].data());
        for (int l = 0; l < L; ++l) iV.emplace_back(env, t_kv, kvb, V[l].data());
        for (int l = 0; l < L; ++l) oK.emplace_back(env, t_kv, kvb);
        for (int l = 0; l < L; ++l) oV.emplace_back(env, t_kv, kvb);
        LiteRtHostBuffer out_log(env, t_log1, kVocab * 4);
        std::vector<LiteRtTensorBuffer> ins(step_->in_l.size());
        for (size_t i = 0; i < ins.size(); ++i) { int n = step_->in_l[i];
            ins[i] = n == 0 ? in_tok.raw() : n == 1 ? in_sp.raw() : n == 2 ? in_cp.raw() : (n - 3 < L ? iK[n - 3].raw() : iV[n - 3 - L].raw()); }
        std::vector<LiteRtTensorBuffer> outs(step_->out_l.size());
        for (size_t j = 0; j < outs.size(); ++j) { int m = step_->out_l[j]; outs[j] = m == 0 ? out_log.raw() : (m <= L ? oK[m - 1].raw() : oV[m - 1 - L].raw()); }
        litert_check(LiteRtRunCompiledModel(step_->compiled, 0, ins.size(), ins.data(), outs.size(), outs.data()), "step");
        out_log.read(logits.data(), kVocab * 4);

        // greedy path (temperature <= 0): pure argmax on RAW logits — matches the
        // verified reference loop and reliably hits the stop token. (Repetition
        // penalty perturbs the clean path and prevents stopping; it belongs with
        // sampling once the alignment-stream-analyzer is ported.)
        if (temperature_ <= 0.f) {
            int best = 0; for (int i = 1; i < kVocab; ++i) if (logits[i] > logits[best]) best = i;
            if (best == kStop) break;
            gen.push_back(best);
            for (int l = 0; l < L; ++l) { oK[l].read(K[l].data(), kvb); oV[l].read(V[l].data(), kvb); }
            tok = best; continue;
        }

        // --- sampling path ---
        // repetition penalty (HF): seen token score/penalty if >0 else *penalty
        if (repetition_penalty_ != 1.f) for (int g : gen) { float& z = logits[g]; z = z > 0 ? z / repetition_penalty_ : z * repetition_penalty_; }
        // temperature
        if (temperature_ != 1.f && temperature_ > 0) for (float& z : logits) z /= temperature_;
        // softmax probs
        float mx = *std::max_element(logits.begin(), logits.end());
        std::vector<float> pr(kVocab); double sum = 0;
        for (int i = 0; i < kVocab; ++i) { pr[i] = std::exp(logits[i] - mx); sum += pr[i]; }
        for (float& p : pr) p = (float)(p / sum);
        // min_p: drop tokens with prob < min_p * max_prob
        if (min_p_ > 0) { float pmax = *std::max_element(pr.begin(), pr.end()), thr = min_p_ * pmax;
            for (int i = 0; i < kVocab; ++i) if (pr[i] < thr) pr[i] = 0.f; }
        // top_p (nucleus): keep smallest set with cumulative prob >= top_p
        if (top_p_ < 1.f) {
            std::vector<int> idx(kVocab); std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return pr[a] > pr[b]; });
            double cum = 0; std::vector<char> keep(kVocab, 0);
            for (int r = 0; r < kVocab; ++r) { keep[idx[r]] = 1; cum += pr[idx[r]]; if (cum >= top_p_) break; }
            for (int i = 0; i < kVocab; ++i) if (!keep[i]) pr[i] = 0.f;
        }
        std::discrete_distribution<int> dist(pr.begin(), pr.end());
        int nxt = dist(rng);
        if (nxt == kStop) break;
        gen.push_back(nxt);
        for (int l = 0; l < L; ++l) { oK[l].read(K[l].data(), kvb); oV[l].read(V[l].data(), kvb); }
        tok = nxt;
    }
    tokens_generated_ = (int)gen.size();
    if (std::getenv("CB_DEBUG")) {
        std::fprintf(stderr, "[cb] prefix_len=%d text_ids(%zu):", prefix_len, text_ids.size());
        for (size_t i = 0; i < text_ids.size() && i < 18; ++i) std::fprintf(stderr, " %ld", text_ids[i]);
        std::fprintf(stderr, "\n[cb] gen(%zu) first:", gen.size());
        for (size_t i = 0; i < gen.size() && i < 12; ++i) std::fprintf(stderr, " %d", gen[i]);
        std::fprintf(stderr, "\n");
    }
    return gen;
}

// ---- flow: speech tokens -> 24 kHz wav ----
std::vector<float> LiteRTChatterboxTts::flow_to_wav(const std::vector<int>& speech_in, std::mt19937& rng) {
    LiteRtEnvironment env = LiteRTEngine::get().env();
    // The flow encoder is fixed-max kPrefillT (512) tokens = prompt_token ++ speech,
    // and Tmel = Ntok*2 must fit kFlowT (1024). Clamp so we never overflow.
    const int max_speech = kPrefillT - (int)prompt_token_.size();
    std::vector<int> speech = speech_in;
    if ((int)speech.size() > max_speech) speech.resize(max_speech);
    const int Ntok = (int)prompt_token_.size() + (int)speech.size();
    const int Tmel = Ntok * 2, mel_out = Tmel - prompt_feat_frames_;
    if (mel_out <= 0) return {};

    // encoder: token_ids -> mu
    std::vector<float> mu((size_t)kMel * kFlowT);
    {
        std::vector<int64_t> ids(kPrefillT, 0);  // encoder fixed-max 512 tokens
        for (int i = 0; i < (int)prompt_token_.size(); ++i) ids[i] = prompt_token_[i];
        for (int i = 0; i < (int)speech.size(); ++i) ids[prompt_token_.size() + i] = speech[i];
        int32_t tlen = Ntok;
        auto t_ids = make_type(kLiteRtElementTypeInt64, {1, kPrefillT});
        auto t_len = make_type(kLiteRtElementTypeInt32, {1});
        auto t_mu = make_type(kLiteRtElementTypeFloat32, {1, kMel, kFlowT});
        auto t_msk = make_type(kLiteRtElementTypeFloat32, {1, 1, kFlowT});
        LiteRtHostBuffer in_ids(env, t_ids, ids.size() * 8, ids.data()), in_len(env, t_len, 4, &tlen);
        LiteRtHostBuffer o_mu(env, t_mu, mu.size() * 4), o_msk(env, t_msk, (size_t)kFlowT * 4);
        std::vector<LiteRtTensorBuffer> ins(enc_->in_l.size()), outs(enc_->out_l.size());
        for (size_t i = 0; i < ins.size(); ++i) ins[i] = enc_->in_l[i] == 0 ? in_ids.raw() : in_len.raw();
        for (size_t j = 0; j < outs.size(); ++j) outs[j] = enc_->out_l[j] == 0 ? o_mu.raw() : o_msk.raw();
        litert_check(LiteRtRunCompiledModel(enc_->compiled, 0, ins.size(), ins.data(), outs.size(), outs.data()), "enc");
        o_mu.read(mu.data(), mu.size() * 4);
    }
    // mask (deterministic) + zero mu tail beyond Tmel
    std::vector<float> mask(kFlowT, 0.f);
    for (int f = 0; f < kFlowT; ++f) mask[f] = f < Tmel ? 1.f : 0.f;
    for (int c = 0; c < kMel; ++c) for (int f = Tmel; f < kFlowT; ++f) mu[(size_t)c * kFlowT + f] = 0.f;
    // conds: first prompt_feat_frames_ = prompt_feat^T, else 0
    std::vector<float> conds((size_t)kMel * kFlowT, 0.f);
    for (int f = 0; f < prompt_feat_frames_; ++f) for (int c = 0; c < kMel; ++c) conds[(size_t)c * kFlowT + f] = prompt_feat_[(size_t)f * kMel + c];
    // x init = randn over valid Tmel
    std::vector<float> x((size_t)kMel * kFlowT, 0.f);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int c = 0; c < kMel; ++c) for (int f = 0; f < Tmel; ++f) x[(size_t)c * kFlowT + f] = nd(rng);

    float ts[11]; for (int i = 0; i <= 10; ++i) ts[i] = 1.f - std::cos((i / 10.f) * 1.57079632679f);
    auto t_b = make_type(kLiteRtElementTypeFloat32, {2, kMel, kFlowT});
    auto t_m = make_type(kLiteRtElementTypeFloat32, {2, 1, kFlowT});
    auto t_t = make_type(kLiteRtElementTypeFloat32, {2});
    auto t_sp = make_type(kLiteRtElementTypeFloat32, {2, kMel});
    std::vector<float> dxdt((size_t)2 * kMel * kFlowT);
    const size_t mt = (size_t)kMel * kFlowT;
    for (int s = 0; s < 10 && !cancelled_.load(); ++s) {
        float t = ts[s], dt = ts[s + 1] - ts[s];
        std::vector<float> xin(2 * mt), min_(2 * kFlowT), muin(2 * mt, 0.f), cin(2 * mt, 0.f), spin(2 * kMel, 0.f);
        std::vector<float> tin = {t, t};
        std::memcpy(&xin[0], x.data(), mt * 4); std::memcpy(&xin[mt], x.data(), mt * 4);
        std::memcpy(&min_[0], mask.data(), kFlowT * 4); std::memcpy(&min_[kFlowT], mask.data(), kFlowT * 4);
        std::memcpy(&muin[0], mu.data(), mt * 4);
        std::memcpy(&cin[0], conds.data(), mt * 4);
        std::memcpy(&spin[0], spks_.data(), (size_t)kMel * 4);
        LiteRtHostBuffer bx(env, t_b, xin.size() * 4, xin.data()), bm(env, t_m, min_.size() * 4, min_.data());
        LiteRtHostBuffer bmu(env, t_b, muin.size() * 4, muin.data()), bt(env, t_t, tin.size() * 4, tin.data());
        LiteRtHostBuffer bsp(env, t_sp, spin.size() * 4, spin.data()), bc(env, t_b, cin.size() * 4, cin.data());
        LiteRtHostBuffer bo(env, t_b, dxdt.size() * 4);
        LiteRtTensorBuffer logical[6] = {bx.raw(), bm.raw(), bmu.raw(), bt.raw(), bsp.raw(), bc.raw()};
        std::vector<LiteRtTensorBuffer> ins(est_->in_l.size());
        for (size_t i = 0; i < ins.size(); ++i) ins[i] = logical[est_->in_l[i]];
        std::vector<LiteRtTensorBuffer> outs(est_->out_l.size(), bo.raw());
        litert_check(LiteRtRunCompiledModel(est_->compiled, 0, ins.size(), ins.data(), outs.size(), outs.data()), "est");
        bo.read(dxdt.data(), dxdt.size() * 4);
        for (size_t i = 0; i < mt; ++i) x[i] += dt * ((1.f + kCfgRate) * dxdt[i] - kCfgRate * dxdt[mt + i]);
        for (int c = 0; c < kMel; ++c) for (int f = Tmel; f < kFlowT; ++f) x[(size_t)c * kFlowT + f] = 0.f;
    }
    // mel = x[:, prompt_feat_frames_:Tmel] padded to kFlowT, hift -> wav
    std::vector<float> melpad((size_t)kMel * kFlowT, 0.f);
    for (int c = 0; c < kMel; ++c) for (int f = 0; f < mel_out; ++f) melpad[(size_t)c * kFlowT + f] = x[(size_t)c * kFlowT + (prompt_feat_frames_ + f)];
    std::vector<float> wav((size_t)kFlowT * kHop);
    {
        auto t_in = make_type(kLiteRtElementTypeFloat32, {1, kMel, kFlowT});
        auto t_w = make_type(kLiteRtElementTypeFloat32, {1, kFlowT * kHop});
        LiteRtHostBuffer b_in(env, t_in, melpad.size() * 4, melpad.data()), b_w(env, t_w, wav.size() * 4);
        std::vector<LiteRtTensorBuffer> ins(hift_->in_l.size(), b_in.raw()), outs(hift_->out_l.size(), b_w.raw());
        litert_check(LiteRtRunCompiledModel(hift_->compiled, 0, ins.size(), ins.data(), outs.size(), outs.data()), "hift");
        b_w.read(wav.data(), wav.size() * 4);
    }
    wav.resize((size_t)mel_out * kHop);
    return wav;
}

void LiteRTChatterboxTts::synthesize(const std::string& text, const std::string& language, TTSChunkCallback on_chunk) {
    cancelled_.store(false);
    seed_used_ = seed_ ? seed_ : std::random_device{}();
    std::mt19937 rng(seed_used_);
    auto ids = tok_->encode(text, language);
    std::vector<long> text_ids; text_ids.reserve(ids.size() + 2);
    text_ids.push_back(kStartText);
    for (int x : ids) text_ids.push_back(x);
    text_ids.push_back(kStopText);
    auto speech = generate_speech_tokens(text_ids, rng);
    if (speech.empty() || cancelled_.load()) { if (on_chunk) on_chunk(nullptr, 0, true); return; }
    auto wav = flow_to_wav(speech, rng);
    if (on_chunk) on_chunk(wav.data(), wav.size(), true);
}

}  // namespace speech_core
