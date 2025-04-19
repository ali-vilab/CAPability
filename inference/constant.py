import os


X_AK = ""
server_url = ""

headers = {
    'Content-Type': 'application/json',
    'X-AK': X_AK,
}


GEMINI_SYSTEM_PROMPT = 'You are a helpful assistant.'
GEMINI_IMAGE_PROMPT = "Please describe the image in detail. Your description should follow these rules:\n"\
    "a) You should describe each object in the image in detail, including its name, number, color, and spatial relationship between objects.\n"\
    "b) You should describe the scene of the image.\n"\
    "c) You should describe the camera angle when shooting this image, such as level angle, high angle, low angle, or dutch angle.\n"\
    "d) You should describe the style of the image, such as realistic, animated, special-effect, old-fashioned and so on.\n"\
    "e) If there are any texts in the image, you should describe the text content.\n"\
    "f) If you know the character in the image, you should tell his or her name.\n"\
    "Directly output your detailed description in a elaborate paragraph, instead of itemizing them in list form. Your description: "
GEMINI_VIDEO_PROMPT = "Please describe the video in detail. Your description should follow these rules:\n"\
    "a) You should describe each events in the video in order, especially focusing on the behavior and action of characters, including people, animals.\n"\
    "b) You should describe each object in the video in detail, including its name, number, color, and spatial relationship between objects.\n"\
    "c) You should describe the scene of the video.\n"\
    "d) You should describe the camera movement when shooting this video, especially the direction, such as pan left, track right, tilt up, boom down, zoom in, dolly out, and so on.\n"\
    "e) You should describe the style of the video, such as realistic, animated, special-effect, old-fashioned and so on.\n"\
    "f) If there are any texts in the video, you should describe the text content.\n"\
    "g) If you know the character in the video, you should tell his or her name.\n"\
    "Directly output your detailed description in a elaborate paragraph, instead of itemizing them in list form. Your description: "


GPT_SYSTEM_PROMPT = 'You are a helpful assistant.'
GPT_IMAGE_PROMPT = "Please describe the image in detail. Your description should follow these rules:\n"\
    "a) You should describe each object in the image in detail, including its name, number, color, and spatial relationship between objects.\n"\
    "b) You should describe the scene of the image.\n"\
    "c) You should describe the camera angle when shooting this image, such as level angle, high angle, low angle, or dutch angle.\n"\
    "d) You should describe the style of the image, such as realistic, animated, special-effect, old-fashioned and so on.\n"\
    "e) If there are any texts in the image, you should describe the text content.\n"\
    "f) If you know the character in the image, you should tell his or her name.\n"\
    "Directly output your detailed description in a elaborate paragraph, instead of itemizing them in list form. Your description: "
GPT_VIDEO_PROMPT = "Given several frames uniformly sampled from a video, please describe the video (not the individual images!) in detail. Your description should follow these rules:\n"\
    "a) You should describe each events in the video in order, especially focusing on the behavior and action of characters, including people, animals.\n"\
    "b) You should describe each object in the video in detail, including its name, number, color, and spatial relationship between objects.\n"\
    "c) You should describe the scene of the video.\n"\
    "d) You should describe the camera movement when shooting this video, especially the direction, such as pan left, track right, tilt up, boom down, zoom in, dolly out, and so on.\n"\
    "e) You should describe the style of the video, such as realistic, animated, special-effect, old-fashioned and so on.\n"\
    "f) If there are any texts in the video, you should describe the text content.\n"\
    "g) If you know the character in the video, you should tell his or her name.\n"\
    "Directly output your detailed description in a elaborate paragraph, instead of itemizing them in list form. Your description: "









model_prompt_dict = {
    'gemini-1.5-pro': {
        'system_prompt': GEMINI_SYSTEM_PROMPT,
        'image_prompt': GEMINI_IMAGE_PROMPT,
        'video_prompt': GEMINI_VIDEO_PROMPT
    },
    'gemini-2.0-flash': {
        'system_prompt': GEMINI_SYSTEM_PROMPT,
        'image_prompt': GEMINI_IMAGE_PROMPT,
        'video_prompt': GEMINI_VIDEO_PROMPT
    },
    'gemini-2.0-pro': {
        'system_prompt': GEMINI_SYSTEM_PROMPT,
        'image_prompt': GEMINI_IMAGE_PROMPT,
        'video_prompt': GEMINI_VIDEO_PROMPT
    },
    'gpt-4o-0806': {
        'system_prompt': GPT_SYSTEM_PROMPT,
        'image_prompt': GPT_IMAGE_PROMPT,
        'video_prompt': GPT_VIDEO_PROMPT
    },
    'gpt-4o-2024-08-06': {
        'system_prompt': GPT_SYSTEM_PROMPT,
        'image_prompt': GPT_IMAGE_PROMPT,
        'video_prompt': GPT_VIDEO_PROMPT
    },
}