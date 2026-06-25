// Chatterbox flow-path LiteRT verification driver.
//
// Validates the flow back-half in C++ against torch golden (deterministic via a
// shared noise z): flow_encoder_v2(prompt_token ++ speech_tokens) -> mu, then the
// CFG Euler CFM loop (estimator, batch-2) -> mel, then hift -> 24 kHz wav.
//
//   chatterbox_flow_check <bundle_dir> <golden_dir> <lang>
//
// Graphs (fp32 v2 set) loaded from bundle_dir; conditioning + golden from golden_dir.
// ai-edge-torch scrambles I/O order, so every tensor is mapped by its "args_N" /
// "output_N" signature-name index.

#include "speech_core/models/litert_engine.h"
#include <litert/c/litert_model.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace speech_core;

namespace {
constexpr int MEL = 80, TMAX = 1024, PROMPT_FEAT = 314, HOP = 480;

std::vector<char> rf(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + p);
    std::streamsize n = f.tellg(); f.seekg(0);
    std::vector<char> b((size_t)n); f.read(b.data(), n); return b;
}
std::vector<float> rf_f32(const std::string& p) {
    auto b = rf(p); std::vector<float> v(b.size() / 4); std::memcpy(v.data(), b.data(), b.size()); return v;
}
std::vector<int32_t> rf_i32(const std::string& p) {
    auto b = rf(p); std::vector<int32_t> v(b.size() / 4); std::memcpy(v.data(), b.data(), b.size()); return v;
}
std::vector<long> json_int_array(const std::string& s, const std::string& key) {
    size_t k = s.find("\"" + key + "\""); if (k == std::string::npos) throw std::runtime_error("no key " + key);
    size_t lb = s.find('[', k), rb = s.find(']', lb);
    std::vector<long> out; std::stringstream ss(s.substr(lb + 1, rb - lb - 1)); std::string t;
    while (std::getline(ss, t, ',')) { size_t a = t.find_first_not_of(" \t\r\n"); if (a != std::string::npos) out.push_back(std::stol(t.substr(a))); }
    return out;
}
int parse_idx(const std::string& nm, const std::string& tag) {
    std::smatch m; std::regex re(tag + R"(_(\d+))");
    if (std::regex_search(nm, m, re)) return std::stoi(m[1]);
    throw std::runtime_error("no " + tag + "_N in " + nm);
}
struct Graph {
    LiteRtModel model = nullptr; LiteRtCompiledModel compiled = nullptr; LiteRtSignature sig = nullptr;
    std::vector<int> in_l, out_l;
    void load(const std::string& p) {
        LiteRTEngine::get().load(p, false, &model, &compiled);
        litert_check(LiteRtGetModelSignature(model, 0, &sig), "sig");
        LiteRtParamIndex ni = 0, no = 0;
        litert_check(LiteRtGetNumSignatureInputs(sig, &ni), "ni");
        litert_check(LiteRtGetNumSignatureOutputs(sig, &no), "no");
        in_l.resize(ni); out_l.resize(no);
        for (LiteRtParamIndex i = 0; i < ni; ++i) { const char* nm; LiteRtGetSignatureInputName(sig, i, &nm); in_l[i] = parse_idx(nm, "args"); }
        for (LiteRtParamIndex i = 0; i < no; ++i) { const char* nm; LiteRtGetSignatureOutputName(sig, i, &nm); out_l[i] = parse_idx(nm, "output"); }
    }
};
double mae(const float* a, const float* b, size_t n) { double s = 0; for (size_t i = 0; i < n; ++i) s += std::fabs((double)a[i] - b[i]); return s / n; }
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <bundle_dir> <golden_dir> <lang>\n", argv[0]); return 2; }
    std::string bdir = argv[1], gdir = argv[2], lang = argv[3];
    LiteRtEnvironment env = LiteRTEngine::get().env();

    // golden + conditioning
    auto jb = rf(gdir + "/" + lang + ".json"); std::string gj(jb.data(), jb.size());
    auto speech = json_int_array(gj, "speech_tokens");
    auto prompt_token = rf_i32(gdir + "/prompt_token.bin");      // [157]
    auto spks = rf_f32(gdir + "/spks.bin");                      // [80]
    auto prompt_feat = rf_f32(gdir + "/prompt_feat.bin");        // [314*80] (frame-major: [f,c])
    auto z = rf_f32(gdir + "/" + lang + "_z.bin");               // [80*Tmel]
    auto gold_mel = rf_f32(gdir + "/" + lang + "_mel.bin");      // [80*(Tmel-314)]
    auto gold_wav = rf_f32(gdir + "/" + lang + "_wav.bin");

    const int Ntok = (int)prompt_token.size() + (int)speech.size();
    const int Tmel = Ntok * 2;
    const int mel_out = Tmel - PROMPT_FEAT;
    std::printf("lang=%s Ntok=%d Tmel=%d mel_out=%d\n", lang.c_str(), Ntok, Tmel, mel_out);

    Graph enc, est, hift;
    enc.load("C:/tmp/cb-litert-out/flow_encoder_v2.tflite");
    est.load(bdir + "/chatterbox-flow-estimator.tflite");
    hift.load(bdir + "/chatterbox-hift.tflite");

    // ---------- flow encoder v2: token_ids -> mu, mask ----------
    std::vector<int64_t> ids(512, 0);
    for (int i = 0; i < (int)prompt_token.size(); ++i) ids[i] = prompt_token[i];
    for (int i = 0; i < (int)speech.size(); ++i) ids[prompt_token.size() + i] = speech[i];
    int32_t tlen = Ntok;
    std::vector<float> mu(MEL * TMAX), mask(1 * TMAX);
    {
        auto t_ids = make_type(kLiteRtElementTypeInt64, {1, 512});
        auto t_len = make_type(kLiteRtElementTypeInt32, {1});
        auto t_mu  = make_type(kLiteRtElementTypeFloat32, {1, MEL, TMAX});
        auto t_msk = make_type(kLiteRtElementTypeFloat32, {1, 1, TMAX});
        LiteRtHostBuffer in_ids(env, t_ids, ids.size() * 8, ids.data());
        LiteRtHostBuffer in_len(env, t_len, 4, &tlen);
        LiteRtHostBuffer o_mu(env, t_mu, mu.size() * 4), o_msk(env, t_msk, mask.size() * 4);
        std::vector<LiteRtTensorBuffer> ins(enc.in_l.size()), outs(enc.out_l.size());
        for (size_t i = 0; i < enc.in_l.size(); ++i) ins[i] = enc.in_l[i] == 0 ? in_ids.raw() : in_len.raw();
        for (size_t j = 0; j < enc.out_l.size(); ++j) outs[j] = enc.out_l[j] == 0 ? o_mu.raw() : o_msk.raw();
        litert_check(LiteRtRunCompiledModel(enc.compiled, 0, ins.size(), ins.data(), outs.size(), outs.data()), "enc Run");
        o_mu.read(mu.data(), mu.size() * 4);
    }
    // mask is deterministic: valid for frames 0..Tmel-1 (encoder mask output is unreliable)
    for (int f = 0; f < TMAX; ++f) mask[f] = (f < Tmel) ? 1.f : 0.f;
    // zero mu beyond Tmel: torch runs the estimator at true length, so the padded
    // tail must be 0 (not encoder garbage) or it leaks via the DiT temporal convs.
    for (int c = 0; c < MEL; ++c) for (int f = Tmel; f < TMAX; ++f) mu[c * TMAX + f] = 0.f;

    // conds [80,TMAX]: first PROMPT_FEAT frames = prompt_feat^T, else 0
    std::vector<float> conds(MEL * TMAX, 0.f);
    for (int f = 0; f < PROMPT_FEAT; ++f)
        for (int c = 0; c < MEL; ++c) conds[c * TMAX + f] = prompt_feat[f * MEL + c];

    // x [80,TMAX] init from z (z is [80,Tmel]); rest 0
    std::vector<float> x(MEL * TMAX, 0.f);
    for (int c = 0; c < MEL; ++c)
        for (int f = 0; f < Tmel; ++f) x[c * TMAX + f] = z[c * Tmel + f];

    // t_span = 1 - cos(linspace(0,1,11) * pi/2)
    float ts[11]; for (int i = 0; i <= 10; ++i) ts[i] = 1.f - std::cos((i / 10.f) * 1.57079632679f);
    const float CFG = 0.7f;

    // ---------- Euler CFM loop (estimator batch-2 = CFG) ----------
    auto t_x   = make_type(kLiteRtElementTypeFloat32, {2, MEL, TMAX});
    auto t_msk = make_type(kLiteRtElementTypeFloat32, {2, 1, TMAX});
    auto t_t   = make_type(kLiteRtElementTypeFloat32, {2});
    auto t_spk = make_type(kLiteRtElementTypeFloat32, {2, MEL});
    auto t_out = make_type(kLiteRtElementTypeFloat32, {2, MEL, TMAX});
    std::vector<float> dxdt(2 * MEL * TMAX);
    for (int s = 0; s < 10; ++s) {
        float t = ts[s], r = ts[s + 1], dt = r - t;
        std::vector<float> xin(2 * MEL * TMAX), min_(2 * TMAX), muin(2 * MEL * TMAX, 0.f), cin(2 * MEL * TMAX, 0.f);
        std::vector<float> tin = {t, t}, spkin(2 * MEL, 0.f);
        std::memcpy(&xin[0], x.data(), MEL * TMAX * 4); std::memcpy(&xin[MEL * TMAX], x.data(), MEL * TMAX * 4);
        std::memcpy(&min_[0], mask.data(), TMAX * 4);  std::memcpy(&min_[TMAX], mask.data(), TMAX * 4);
        std::memcpy(&muin[0], mu.data(), MEL * TMAX * 4);     // uncond mu stays 0
        std::memcpy(&cin[0], conds.data(), MEL * TMAX * 4);   // uncond cond stays 0
        std::memcpy(&spkin[0], spks.data(), MEL * 4);         // uncond spks stays 0

        LiteRtHostBuffer b_x(env, t_x, xin.size() * 4, xin.data()), b_m(env, t_msk, min_.size() * 4, min_.data());
        LiteRtHostBuffer b_mu(env, t_x, muin.size() * 4, muin.data()), b_t(env, t_t, tin.size() * 4, tin.data());
        LiteRtHostBuffer b_spk(env, t_spk, spkin.size() * 4, spkin.data()), b_c(env, t_x, cin.size() * 4, cin.data());
        LiteRtHostBuffer b_out(env, t_out, dxdt.size() * 4);
        // logical: 0=x,1=mask,2=mu,3=t,4=spks,5=cond
        LiteRtTensorBuffer logical[6] = {b_x.raw(), b_m.raw(), b_mu.raw(), b_t.raw(), b_spk.raw(), b_c.raw()};
        std::vector<LiteRtTensorBuffer> ins(est.in_l.size());
        for (size_t i = 0; i < est.in_l.size(); ++i) ins[i] = logical[est.in_l[i]];
        std::vector<LiteRtTensorBuffer> outs(est.out_l.size(), b_out.raw());
        litert_check(LiteRtRunCompiledModel(est.compiled, 0, ins.size(), ins.data(), outs.size(), outs.data()), "est Run");
        b_out.read(dxdt.data(), dxdt.size() * 4);
        for (int i = 0; i < MEL * TMAX; ++i) {
            float d = (1.f + CFG) * dxdt[i] - CFG * dxdt[MEL * TMAX + i];
            x[i] += dt * d;
        }
        // keep the padded tail at exactly 0 (match torch's true-length extent)
        for (int c = 0; c < MEL; ++c) for (int f = Tmel; f < TMAX; ++f) x[c * TMAX + f] = 0.f;
        if (s == 0) { double r2 = 0; for (int i = 0; i < MEL * TMAX; ++i) r2 += (double)dxdt[i] * dxdt[i];
            std::printf("DBG step0 dxdt_rms=%.4f t=%.4f r=%.4f\n", std::sqrt(r2 / (MEL * TMAX)), t, r); }
    }

    // mel = x[:, PROMPT_FEAT:Tmel] -> [80, mel_out]
    std::vector<float> mel(MEL * mel_out);
    for (int c = 0; c < MEL; ++c)
        for (int f = 0; f < mel_out; ++f) mel[c * mel_out + f] = x[c * TMAX + (PROMPT_FEAT + f)];
    double mel_mae = mae(mel.data(), gold_mel.data(), std::min(mel.size(), gold_mel.size()));
    std::printf("mel: c++=%zu golden=%zu MAE=%.6f\n", mel.size(), gold_mel.size(), mel_mae);

    // ---------- hift: mel[1,80,TMAX] (padded) -> wav ----------
    std::vector<float> melpad(MEL * TMAX, 0.f);
    for (int c = 0; c < MEL; ++c) for (int f = 0; f < mel_out; ++f) melpad[c * TMAX + f] = mel[c * mel_out + f];
    std::vector<float> wav(TMAX * HOP);
    {
        auto t_in = make_type(kLiteRtElementTypeFloat32, {1, MEL, TMAX});
        LiteRtHostBuffer b_in(env, t_in, melpad.size() * 4, melpad.data());
        // hift output length = TMAX*HOP; query via output tensor not needed, size known
        auto t_w = make_type(kLiteRtElementTypeFloat32, {1, TMAX * HOP});
        LiteRtHostBuffer b_w(env, t_w, wav.size() * 4);
        std::vector<LiteRtTensorBuffer> ins(hift.in_l.size(), b_in.raw());
        std::vector<LiteRtTensorBuffer> outs(hift.out_l.size(), b_w.raw());
        litert_check(LiteRtRunCompiledModel(hift.compiled, 0, ins.size(), ins.data(), outs.size(), outs.data()), "hift Run");
        b_w.read(wav.data(), wav.size() * 4);
    }
    int wav_n = mel_out * HOP;
    double wav_mae = mae(wav.data(), gold_wav.data(), std::min((size_t)wav_n, gold_wav.size()));
    std::printf("wav: c++=%d golden=%zu MAE=%.6f\n", wav_n, gold_wav.size(), wav_mae);

    // dump C++ wav (raw f32 @ 24 kHz) for Whisper WER comparison vs the torch reference
    { std::ofstream o(gdir + "/" + lang + "_cpp_wav.bin", std::ios::binary);
      o.write((const char*)wav.data(), (size_t)wav_n * 4); }

    bool ok = wav_mae < 0.03;  // padding-approximation tolerance (WER is the real metric)
    std::printf(ok ? "FLOW-CHECK-PASS\n" : "FLOW-CHECK-FAIL\n");
    return ok ? 0 : 1;
}
