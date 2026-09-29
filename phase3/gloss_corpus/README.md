
Spanish-MSL glosses dataset -IT is a .xlsx format file that contains 3000 Spanish-MSL glosses pairs. To use dataset it needs to be converted to .csv format

Model M1- It is a Colab file that contains the programming methodology for finetunning Helsinki-NLP/opus-mt-es-es available on Hugging Face: https://huggingface.co/Helsinki-NLP/opus-mt-es-es. 
It was fine-tuned on MSL-Spanish glosses corpus. It uses transformers library from Hugging Face, the trainer API for translation and evaluation. To evaluate the quality of translation, ROUGE, TER, BLEU were measured.
The model card of M1 is available at: https://huggingface.co/VaniLara/esp-to-lsm-model, there is a guide on how to use it on the transformers library.
If you want to use the Colab you will need to create an access token, use your own google drive account and create a repo on hugging face.

Model M2- It is a Colab file that contains the programming methodology for finetunning vgaraujov/bart-base-spanish  available on Hugging Face: https://huggingface.co/Helsinki-NLP/opus-mt-es-es. 
It was fine-tuned on MSL-Spanish glosses corpus. It uses transformers library from Hugging Face, the trainer API for translation and evaluation. To evaluate the quality of translation, ROUGE, TER, BLEU were measured.
The model card of M1 is available at: [https://huggingface.co/VaniLara/esp-to-lsm-barto-model](https://huggingface.co/VaniLara/esp-to-lsm-model), there is a guide on how to use it on the transformers library.
If you want to use the Colab you will need to create an access token, use your own google drive account and create a repo on hugging face.

Model M1-split-version and Model M2-split-version is the dataset splitted in 80% training, 10% validation and 10% testing. Model cards are avilable at: https://huggingface.co/vania2911/esp-to-lsm-barto-model and https://huggingface.co/vania2911/esp-to-lsm-model-split.

Translations M1 and M2 contain the reference and predicted translations for each model.
