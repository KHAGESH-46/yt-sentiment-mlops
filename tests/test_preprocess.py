from src.prepare import clean_text, weak_label


def test_clean_text_strips_urls_and_whitespace():
    assert clean_text("Check https://x.com/abc   out!!") == "check out!!"


def test_clean_text_keeps_emoji():
    assert "🔥" in clean_text("Great video 🔥")


def test_clean_text_lowercases():
    assert clean_text("AMAZING Video") == "amazing video"


def test_weak_label_positive():
    assert weak_label("this is great, love it 😍") == "positive"


def test_weak_label_negative():
    assert weak_label("worst video ever, trash 💀") == "negative"


def test_weak_label_neutral():
    assert weak_label("who is watching in 2026?") == "neutral"
