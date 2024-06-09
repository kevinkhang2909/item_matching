This repository contains the code for a data cleaning and image-text matching workflow.

# 1. Data Cleaning
The workflow consists of two main functions for cleaning text and image data:

## PipelineImage: 
This function performs the following steps on image data:
- Downloads images using the `img2dataset` or `requests` library.
- Validates downloaded images for errors or existing files.
- Passes the extracted text from images to the PipelineText function for cleaning.

Running the Cleaning Pipeline:
```
PipelineImage().run()
```

## PipelineText: 
This function performs basic text cleaning using regular expressions:
- Converts all text to lowercase.
- Removes leading and trailing whitespace.

Running the Cleaning Pipeline:
```
PipelineText().run()
```

# 2. Model Selection
The workflow supports different pre-trained models for text and image data:
- Text: `BAAI/bge-m3`
- Images: `google/siglip-base-patch16-22`

Running the Model:
```
Model().get_text_model()
Model().get_img_model()
```

**Note:** These are just example models. You can customize this section to specify the models you are using.

# 3. Matching
This section outlines the steps for large-scale matching between cleaned text and image data. The specific implementation details might be subject to change.

- **Load Clean Data:** Load the cleaned text and image data from the previous step.
- **Convert to Embeddings:** Convert both text and image data into vector representations (embeddings) using chunking with the `Datasets` library.
- **Build Index:** Build an efficient index for fast retrieval using the `autofaiss` library.
- **Query:** Perform text queries on the image embeddings using the `Datasets` library.

Running the Matching:
```
BELargeScale().match()
```
