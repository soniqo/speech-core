#include "speech_core/models/nemotron_language_guide.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <limits>
#include <numeric>

int main() {
    const std::unordered_map<std::string, int> languages{
        {"auto",101},{"en-US",0},{"en-GB",1},{"ru-RU",11},{"zh-CN",4},
        {"en",0},{"reserved",80},{"bad",128}};
    const std::vector<std::string> vocab{"<en-US>","<en-GB>","<ru-RU>","<zh-CN>","<bad>"};
    auto mask = [&](std::vector<std::string> expected, float weight=.1f, int selected=101) {
        return speech_core::nemotron_language_guide_mask(expected,weight,128,101,selected,languages,vocab);
    };
    assert(mask({}).empty());
    assert(mask({"en-US"},0).empty());
    const auto a=mask({"en-US","ru-RU"});
    assert(a.size()==128 && a[101]==.9f && a[0]==.05f && a[11]==.05f);
    assert(std::fabs(std::accumulate(a.begin(),a.end(),0.f)-1.f)<1e-6f);
    assert(mask({"en-US","en-US","ru-RU"})==a);
    auto b=mask({"zh-CN"});
    assert(b[4]==.1f && b[0]==0 && a[4]==0); // Independent session values.
    auto rejects = [&](std::vector<std::string> expected,float weight=.1f,int selected=101) {
        try { (void)mask(expected,weight,selected); return false; }
        catch (const std::invalid_argument&) { return true; }
    };
    assert(rejects({"en"}) && rejects({"auto"}) && rejects({"reserved"}) && rejects({"bad"}));
    assert(rejects({"en-US","unknown"}) && rejects({" en-US"}));
    assert(rejects({"en-US"},.1f,0) && rejects({"en-US"},0,0));
    assert(rejects({},-.1f) && rejects({},.11f));
    assert(rejects({},std::numeric_limits<float>::quiet_NaN()));
    assert(rejects({},std::numeric_limits<float>::infinity()));
    assert(a==mask({"en-US","ru-RU"})); // Rejected configurations changed nothing.
    return 0;
}
