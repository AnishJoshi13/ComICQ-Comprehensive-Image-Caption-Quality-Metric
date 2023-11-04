from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
from pycocotools.coco import COCO
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Base path for the images
base_image_dir = 'C:/pythonProject/CVCP/image_captioning_with_transformers/dataset_dir/val2017/'

# Specify the image filename
image_filename = '000000000776.jpg'

# List to store generated captions
generated_captions = []

# Generate captions for the specified image
image_path = os.path.join(base_image_dir, image_filename)

if os.path.isfile(image_path):
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")

        # Generate image caption
        with torch.no_grad():
            caption = model.generate(**inputs)

        # Decode and add the caption to the list
        caption_text = processor.decode(caption[0], skip_special_tokens=True)
        generated_captions.append({
            'image_filename': image_filename,
            'caption': caption_text
        })
    except FileNotFoundError:
        print(f"Image not found: {image_path}")

# Load the COCO annotation file (val set)
annotation_file = 'captions_val2017.json'  # Adjust the path accordingly
coco = COCO(annotation_file)

# Initialize a list to store reference captions for the specified image
image_id = 776  # Manually specify the image ID for '000000000776.jpg'
ann_ids = coco.getAnnIds(imgIds=image_id)
reference_captions = [ann['caption'] for ann in coco.loadAnns(ann_ids)]

# Initialize scorers
bleu_scorer = Bleu()
rouge_scorer = Rouge()
cider_scorer = Cider()
meteor_scorer = Meteor()

# Prepare data for scoring
references = {image_id: reference_captions}
hypotheses = {image_id: [generated_captions[0]['caption']]}

# Compute scores for the specified image
bleu_scores, _ = bleu_scorer.compute_score(references, hypotheses)
rouge_scores, _ = rouge_scorer.compute_score(references, hypotheses)
cider_scores, _ = cider_scorer.compute_score(references, hypotheses)
meteor_scores, _ = meteor_scorer.compute_score(references, hypotheses)

# Print the generated caption and reference captions for the specified image
print("Image Filename:", generated_captions[0]['image_filename'])
print("Generated Caption:", generated_captions[0]['caption'])

print("Reference Captions:")
for i, ref_caption in enumerate(reference_captions):
    print(f"Reference {i+1}: {ref_caption}")

print("BLEU Scores:", bleu_scores)
print("ROUGE Scores:", rouge_scores)
print("CIDEr Scores:", cider_scores)
print("METEOR Scores:", meteor_scores)




# from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image
# import torch
# import os
# from pycocotools.coco import COCO
# from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor
#
# # Load the BLIP processor and model
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#
# # Base path for the images
# base_image_dir = 'C:/pythonProject/CVCP/image_captioning_with_transformers/dataset_dir/val2017/'
#
# # Specify the image filenames (without the base path)
# image_filenames = [
#     '000000000139.jpg',
#     '000000000285.jpg',
#     '000000000632.jpg',
#     '000000000724.jpg',
#     '000000000776.jpg'
# ]
#
# # List to store generated captions
# generated_captions = []
#
# # Generate captions for each image
# for image_filename in image_filenames:
#     image_path = os.path.join(base_image_dir, image_filename)
#
#     if os.path.isfile(image_path):
#         try:
#             image = Image.open(image_path)
#             inputs = processor(images=image, return_tensors="pt")
#
#             # Generate image caption
#             with torch.no_grad():
#                 caption = model.generate(**inputs)
#
#             # Decode and add the caption to the list
#             caption_text = processor.decode(caption[0], skip_special_tokens=True)
#             generated_captions.append({
#                 'image_filename': image_filename,
#                 'caption': caption_text
#             })
#         except FileNotFoundError:
#             print(f"Image not found: {image_path}")
#             continue
#
# # Load the COCO annotation file (val set)
# annotation_file = 'captions_val2017.json'  # Adjust the path accordingly
# coco = COCO(annotation_file)
#
# # Initialize a list to store reference captions
# reference_captions = []
# image_ids = coco.getImgIds()
#
# # Specify the image IDs for the 5 images
# selected_image_ids = [139, 285, 632, 724, 776]
#
# for image_id in selected_image_ids:
#     ann_ids = coco.getAnnIds(imgIds=image_id)
#     captions = [ann['caption'] for ann in coco.loadAnns(ann_ids)]
#     reference_captions.append({
#         'image_id': image_id,
#         'caption': captions
#     })
#
# # Initialize scorers
# bleu_scorer = Bleu()
# rouge_scorer = Rouge()
# cider_scorer = Cider()
# meteor_scorer = Meteor()
#
# # Prepare data for scoring
# references = {}
# hypotheses = {}
#
# for ref in reference_captions:
#     image_id = ref['image_id']
#     references[image_id] = ref['caption']
#
# for gen in generated_captions:
#     image_filename = gen['image_filename']
#     image_id = int(os.path.splitext(image_filename)[0])
#     hypotheses[image_id] = [gen['caption']]
#
# # Compute scores
# bleu_scores, _ = bleu_scorer.compute_score(references, hypotheses)
# rouge_scores, _ = rouge_scorer.compute_score(references, hypotheses)
# cider_scores, _ = cider_scorer.compute_score(references, hypotheses)
# meteor_scores, _ = meteor_scorer.compute_score(references, hypotheses)
#
# # Print the generated captions and scores
# for caption_info in generated_captions:
#     print("Image Filename:", caption_info['image_filename'])
#     print("Generated Caption:", caption_info['caption'])
#
#     # Find the corresponding reference captions for this image
#     image_id = int(os.path.splitext(caption_info['image_filename'])[0])
#     ref_captions = reference_captions[selected_image_ids.index(image_id)]['caption']
#     print("Reference Captions:")
#     for i, ref_caption in enumerate(ref_captions):
#         print(f"Reference {i+1}: {ref_caption}")
#
#     print()
#
# print("BLEU Scores:", bleu_scores)
# print("ROUGE Scores:", rouge_scores)
# print("CIDEr Scores:", cider_scores)
# print("METEOR Scores:", meteor_scores)
