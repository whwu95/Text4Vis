import torch
import clip

def text_prompt(data):
    # text_aug = ['{}']
    text_aug = ['a video of a person {}.']

    # Kinetics
    # text_aug = [
    #     'a photo of a person {}.',
    #     'a photo of {}.',
    #     'a photo of a person using {}.',
    #     'a photo of a person doing {}.',
    #     'a photo of a person during {}.',
    #     'a photo of a person performing {}.',
    #     'a photo of a person practicing {}.',
    #     'a video of {}.',
    #     'a video of a person {}.',
    #     'a video of a person using {}.',
    #     'a video of a person doing {}.',
    #     'a video of a person during {}.',
    #     'a video of a person performing {}.',
    #     'a video of a person practicing {}.',
    #     'a example of {}.',
    #     'a example of a person {}.',
    #     'a example of a person using {}.',
    #     'a example of a person doing {}.',
    #     'a example of a person during {}.',
    #     'a example of a person performing {}.',
    #     'a example of a person practicing {}.',
    #     'a demonstration of {}.',
    #     'a demonstration of a person {}.',
    #     'a demonstration of a person using {}.',
    #     'a demonstration of a person doing {}.',
    #     'a demonstration of a person during {}.',
    #     'a demonstration of a person performing {}.',
    #     'a demonstration of a person practicing {}.',
    # ]

    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])

    classes = text_dict[0] 

    return classes, num_text_aug, text_dict