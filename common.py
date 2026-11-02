import re

MODEL = "Qwen/Qwen3-ASR-1.7B"
PROVIDERS = {
    "siliconflow": {"name": "硅基流动", "service": "ASR-Web-SiliconFlow", "key_pattern": r"sk-[A-Za-z0-9_-]{20,250}"},
    "deepinfra": {"name": "DeepInfra", "service": "ASR-Web-DeepInfra", "key_pattern": r"[A-Za-z0-9_.-]{20,250}"},
}
ASR_ENGINES = {
    "siliconflow": {"provider": "siliconflow", "model": "Qwen/Qwen3-ASR-1.7B",
                    "label": "云端 · 硅基流动 Qwen3-ASR-1.7B", "price_per_second": 0.00022},
    "deepinfra": {"provider": "deepinfra", "model": "Qwen/Qwen3-ASR-1.7B",
                  "label": "云端 · DeepInfra Qwen3-ASR-1.7B（更便宜）", "price_per_second": 0.0000075 * 7.2},
    "local-1.7b": {"provider": "local", "model": "Qwen3-ASR-1.7B-hf", "vram_mb": 6000,
                   "label": "本地 GPU · Qwen3-ASR-1.7B（免费，需约 6GB 空闲显存）", "price_per_second": 0},
    "local-0.6b": {"provider": "local", "model": "Qwen3-ASR-0.6B-hf", "vram_mb": 3000,
                   "label": "本地 GPU · Qwen3-ASR-0.6B（免费，需约 3GB 空闲显存）", "price_per_second": 0},
}
DEFAULT_ASR = "siliconflow"
TRANSLATION_MODELS = {
    "pro": {"provider": "siliconflow", "model": "deepseek-ai/DeepSeek-V4-Pro", "label": "硅基流动 · DeepSeek-V4-Pro（推荐）"},
    "flash": {"provider": "siliconflow", "model": "deepseek-ai/DeepSeek-V4.1-Flash", "label": "硅基流动 · DeepSeek-V4.1-Flash（快速）"},
    "deepinfra-pro": {"provider": "deepinfra", "model": "deepseek-ai/DeepSeek-V4-Pro", "label": "DeepInfra · DeepSeek-V4-Pro"},
    "deepinfra-flash": {"provider": "deepinfra", "model": "deepseek-ai/DeepSeek-V4.1-Flash", "label": "DeepInfra · DeepSeek-V4.1-Flash（快速）"},
}
DEFAULT_TRANSLATION = "pro"
TRANSLATION_MODEL = TRANSLATION_MODELS[DEFAULT_TRANSLATION]["model"]
SOURCE_LANGUAGES = {
    "ja": "日语",
    "zh": "中文",
    "en": "英语",
    "ko": "韩语",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "ru": "俄语",
    "auto": "自动识别（不过滤）",
}
_LATIN = r"[A-Za-z\u00c0-\u024f]"
_SCRIPTS = {
    "ja": r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]",
    "zh": r"[\u3400-\u4dbf\u4e00-\u9fff]",
    "ko": r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]",
    "en": _LATIN, "fr": _LATIN, "de": _LATIN, "es": _LATIN,
    "ru": r"[\u0400-\u04ff]",
}
# Pure moans, gasps and breathing written in kana; short replies with meaning are kept.
_NONVERBAL = set("あぁアァんンはハふフうぅウゥおぉオォえぇエェひヒ")
_IGNORABLE = set("っッーｰ〜～~…・、。，,.．!！?？「」『』（）()　 ♡♥♪-")
_MEANINGFUL = {"うん", "ううん", "ええ", "うんうん", "ウン", "エエ"}


def language_matches(text, source_language):
    pattern = _SCRIPTS.get(source_language)
    return pattern is None or re.search(pattern, text) is not None


def is_nonverbal(text):
    core = "".join(c for c in text if c not in _IGNORABLE and not c.isspace())
    if core in _MEANINGFUL:
        return False
    return not core or set(core) <= _NONVERBAL
LANGUAGES = {
    "original": "原文（不翻译）",
    "zh-CN": "简体中文",
    "zh-TW": "繁体中文",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "pt": "葡萄牙语",
    "it": "意大利语",
    "ru": "俄语",
    "vi": "越南语",
    "th": "泰语",
    "id": "印度尼西亚语",
    "ar": "阿拉伯语",
}


class UserError(Exception):
    pass


class Cancelled(UserError):
    pass


def safe_message(value, api_key=""):
    text = str(value)
    if api_key:
        text = text.replace(api_key, "[API Key 已隐藏]")
    return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[API Key 已隐藏]", text)[:800]


def check_cancel(event):
    if event.is_set():
        raise Cancelled("已取消，已发出的接口请求可能仍计费；未发送的片段不再提交。")
