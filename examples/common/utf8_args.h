#pragma once
// UTF-8 argv for example CLIs.
//
// MSVC's `char** argv` is converted from the wide command line through the
// active code page (e.g. Windows-1252), which cannot represent most
// non-Latin scripts — a Devanagari text argument arrives as a row of '?'
// and the model is asked to speak question marks. Rebuild the argument
// vector from GetCommandLineW() as UTF-8 instead. On other platforms argv
// is already UTF-8 and is passed through unchanged.

#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace speech_examples {

inline std::vector<std::string> utf8_args(int argc, char** argv) {
#if defined(_WIN32)
    int wargc = 0;
    if (wchar_t** wargv = CommandLineToArgvW(GetCommandLineW(), &wargc)) {
        std::vector<std::string> args;
        args.reserve(static_cast<size_t>(wargc));
        for (int i = 0; i < wargc; ++i) {
            const int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                              nullptr, 0, nullptr, nullptr);
            std::string s;
            if (n > 1) {
                s.resize(static_cast<size_t>(n) - 1);
                WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), n,
                                    nullptr, nullptr);
            }
            args.push_back(std::move(s));
        }
        LocalFree(wargv);
        if (!args.empty()) return args;
    }
#endif
    return std::vector<std::string>(argv, argv + argc);
}

}  // namespace speech_examples
