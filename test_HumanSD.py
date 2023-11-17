from modelscope.models import Model
from modelscope.pipelines import pipeline
import ms_wrapper
import os
model = "damo/Pose-driven-image-generation-HumanSD"
image_dir = "fff1daa9ef182b86387d802dba686426e7feb396_19692096.jpg"
prompt = "a woman"
neg_prompt = "lmonochrome, lowres, bad anatomy, worst quality, low quality"
inference = pipeline('Pose-driven-image-generation', model = model,model_revision="v1.0.1")
inputs = {"image_dir":image_dir,"prompt":prompt,"neg_prompt": neg_prompt,"sample_steps":30,"seed":None,"guidance_scale":10.0,"num_samples":2 }
output = inference(input=inputs)
output_dir = 'test'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
for i in range(len(output)):
    output[i].save(os.path.join(output_dir,str(i).zfill(4)+'.jpg'))