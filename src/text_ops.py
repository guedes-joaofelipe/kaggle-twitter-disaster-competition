import re


def clean_keyword(keyword: str) -> str:
    if not isinstance(keyword, str):
        return ""

    return keyword.replace("%20", " ").lower()


def clean_text(text: str) -> str:
    return text.replace("%20", " ").lower()


def get_text_tags(text: str, regex=r"#") -> list:
    return re.findall(regex, text)


# transformers = {
#     "clean_keyword": FunctionTransformer(clean_keyword),
#     "get_profile_tags": FunctionTransformer(lambda x: get_text_tags(x, r"@\w+")),
#     "get_hash_tags": FunctionTransformer(lambda x: get_text_tags(x, r"#\w+")),
#     "get_link_tags": FunctionTransformer(
#         lambda x: get_text_tags(x, r"https://t.co/\w+")
#     ),
# }
