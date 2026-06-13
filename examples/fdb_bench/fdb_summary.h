// Public surface of fdb_summary so the smoke test can drive the same
// implementation without spawning a process. Vendored under examples/
// (not in include/speech_core/) because the post-processor is a tool,
// not a library API.

#pragma once

#include <ostream>
#include <string>

namespace fdb_summary {

/// Read every *.json file under `in_dir`, bucket samples by
/// (category, stt_backend, tts_backend, llm_model), and write the
/// summary CSV to `csv`. Returns true on success. Skips unparseable
/// files with a warning to stderr; returns false only on argument
/// errors (e.g. in_dir does not exist).
bool process_dir(const std::string& in_dir, std::ostream& csv);

}  // namespace fdb_summary
