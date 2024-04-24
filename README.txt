Directory structure:

Put DeepHS and ecostress into their own directories
DeepHS fruit dirs should be unzipped and placed in DeepHS.
Create a folder for models to save and load

|-DeepHS/
|--annotations
|--Avacado
|--Kaki
|--Kiwi
|--Mango
|--Papaya
|-ecospeclib-all/
|-models/
|-net/
|-main.ibynb

Optionally, swap out DeepHS for any HSI images. Create utils and sequence files to match then run the VAEs.