# Voice Activity Projection (ONNX)

`OnnxVap` is an optional stereo forecasting model in `speech_core_models`.
Build with `SPEECH_CORE_WITH_ONNX=ON` and provide ONNX Runtime through `ORT_DIR`.

```cpp
#include <speech_core/models/onnx_vap.h>

speech_core::OnnxVap model("/models/vap.onnx");
auto forecast = model.forecast(user.data(), agent.data(), user.size(), 16000);
// Each pair is [user, agent].
auto near_term = forecast.p_now;
auto later = forecast.p_future;
auto speaking = forecast.voice_activity;
```

Both arrays must contain the requested number of synchronized 16 kHz samples.
Supply zeros for an inactive agent. Each request uses the latest 20 seconds
and prepends silence when shorter. The graph consumes channel-major
`audio [1, 2, 320000]` Float32 and returns `p_now`, `p_future` and `vad`, each
`[1, 1000, 2]`. The wrapper returns their last frame and checks the graph's
names, types and dimensions when loading.

Serialize calls to an instance. Run the request on a worker thread; each call
recomputes the full context. The application chooses its response and
interruption policy from the forecasts. This optional model leaves existing
pipeline defaults in place.

The fixed-window export was compared with the original checkpoint on silence,
separate user/agent signals and simultaneous signals. Native tests also pin
the channel order, leading padding, recent-context selection and reference
probabilities. Set `SPEECH_VAP_ONNX` to the downloaded graph before running
`ctest -R test_onnx_vap_model --output-on-failure`.

Weights: [viks66/VAP_checkpoints](https://huggingface.co/viks66/VAP_checkpoints),
`oto.ckpt` at `b9aa0ba1718221153e04ef343c7f3b8cf84bfbc3`, CC BY 4.0.
Architecture: [VoiceActivityProjection](https://github.com/ErikEkstedt/VoiceActivityProjection),
MIT. Keep the accompanying attribution with redistributed weights.
