from vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os
dataDir ='C:/Users/rlowande/Desktop/test1'
versionType =''
taskType ='OpenEnded'
dataType ='mscoco'
dataSubType ='train2014'
annFile ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType,
dataType, dataSubType)
quesFile ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType,
taskType, dataType, dataSubType)
imgDir = '%s/Images/train2014/' %(dataDir)
# initialize VQA api for QA annotations
vqa=VQA(annFile, quesFile)
# load and display QA annotations for given question types
annIds = vqa.getQuesIds(quesTypes='how many');
anns = vqa.loadQA(annIds)
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])
imgId = randomAnn['image_id']
imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
if os.path.isfile(imgDir + imgFilename):

I = io.imread(imgDir + imgFilename)
plt.imshow(I)
plt.axis('off')
plt.show()
# load and display QA annotations for given answer types
annIds = vqa.getQuesIds(ansTypes='yes/no');
anns = vqa.loadQA(annIds)
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])
imgId = randomAnn['image_id']
imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
if os.path.isfile(imgDir + imgFilename):
I = io.imread(imgDir + imgFilename)
plt.imshow(I)
plt.axis('off')
plt.show()
# load and display QA annotations for given images
ids = vqa.getImgIds()
annIds = vqa.getQuesIds(imgIds=random.sample(ids,5));
anns = vqa.loadQA(annIds)
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])
imgId = randomAnn['image_id']
imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
if os.path.isfile(imgDir + imgFilename):
I = io.imread(imgDir + imgFilename)
plt.imshow(I)
plt.axis('off')
plt.show()
