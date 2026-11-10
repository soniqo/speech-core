# Experimental Nemotron language guide

The multilingual ONNX and LiteRT recognizers accept `set_language_guide`
before `begin_stream`. Select `auto` first, then pass exact locale keys such
as `{"en-US", "ru-RU", "zh-CN"}`. Each key must occur in both the loaded
prompt dictionary and language-tag vocabulary. Unknown, aliased, reserved or
out-of-range keys fail without replacing the current guide.

At the default weight, the prompt mask retains 90% on Auto and divides 10%
equally across distinct selected prompt slots. An empty list or zero weight
uses the original one-hot path. Weights must be finite and between 0 and 0.1.
Every `set_language` call clears the guide, including selecting Auto again.
`set_language_guide` rejects an open stream; cancel or finish it first.

This is experimental fractional conditioning through a nonlinear prompt
projector, outside the model's documented one-hot input contract. It does not
mask decoder vocabulary, exclude other languages or guarantee improved
recognition. Word times, confidence computation and decoder behavior are
otherwise unchanged. Guide masks belong to individual instances; an ONNX
stream clone copies the configuration and starts with independent caches.
