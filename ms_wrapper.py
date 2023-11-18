# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import torch
import numpy as np
import cv2
import seaborn as sns
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler as humansd_DDIMSampler
from mmpose.apis import inference_bottom_up_pose_model, init_pose_model
from modelscope.models.base import TorchModel
from modelscope.preprocessors.base import Preprocessor
from modelscope.pipelines.base import Model, Pipeline
from modelscope.utils.config import Config
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.models.builder import MODELS
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
IMAGE_RESOLUTION=512

@MODELS.register_module('Pose-driven-image-generation', module_name='HumanSD')
class MyCustomModel(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.gen_model,self.sampler,self.pose_model = self.init_model(model_dir,**kwargs)
        
    def forward(self, image_dir,prompt,neg_prompt,sample_steps,seed,guidance_scale,num_samples, **forward_params):
        input_image = Image.open(image_dir)
        image = np.array(input_image.convert("RGB"))
        image = self.resize_image(image,IMAGE_RESOLUTION) 
        mmpose_results=inference_bottom_up_pose_model(self.pose_model, image, dataset='BottomUpCocoDataset', dataset_info=None, pose_nms_thr=1.0, return_heatmap=False, outputs=None)[0]
        mmpose_filtered_results=[]
        for mmpose_result in mmpose_results:
            if mmpose_result["score"]>0.01:
                mmpose_filtered_results.append(mmpose_result)
                
        humansd_pose_image=self.draw_humansd_skeleton(image,mmpose_filtered_results,0.01)
        
        # humansd
        self.sampler.make_schedule(sample_steps, ddim_eta=0.0, verbose=True)
        do_full_sample = 1.0 == 1.
            
        t_enc = min(int(1.0 * sample_steps), sample_steps-1)
        humansd_result = self.paint_humansd(
            humansd_sampler=self.sampler,
            image=image,
            pose_image=humansd_pose_image,
            prompt=prompt,
            t_enc=t_enc,
            seed=seed,
            scale=guidance_scale,
            num_samples=num_samples,
            callback=None,
            do_full_sample=do_full_sample,
            device=DEVICE,
            negative_prompt=neg_prompt
        )
        # save_image=np.concatenate(humansd_result,0)
                
        # save_image=save_image[...,[2,1,0]]
        return humansd_result

    def init_model(self, model_dir,**kwargs):
        """Provide default implementation based on TorchModel and user can reimplement it.
            include init model and load ckpt from the model_dir, maybe include preprocessor
            if nothing to do, then return lambda x: x
        """
        humansd_config=OmegaConf.load("configs/humansd/humansd-inference.yaml")
        humansd_model = instantiate_from_config(humansd_config.model)
        humansd_model.load_state_dict(torch.load(os.path.join(model_dir,"humansd-v1.ckpt"))["state_dict"], strict=False)
        humansd_model = humansd_model.to(DEVICE)
        humansd_sampler = humansd_DDIMSampler(humansd_model)
        
        mmpose_model=init_pose_model("humansd_data/models/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512_udp.py", os.path.join(model_dir,"higherhrnet_w48_humanart_512x512_udp.pth"), device=DEVICE)
        return humansd_model,humansd_sampler,mmpose_model

    def resize_image(self,input_image, resolution):
        H, W, C = input_image.shape
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img
    
    def draw_humansd_skeleton(self,image, present_pose,mmpose_detection_thresh):
        humansd_skeleton=[
                [0,0,1],
                [1,0,2],
                [2,1,3],
                [3,2,4],
                [4,3,5],
                [5,4,6],
                [6,5,7],
                [7,6,8],
                [8,7,9],
                [9,8,10],
                [10,5,11],
                [11,6,12],
                [12,11,13],
                [13,12,14],
                [14,13,15],
                [15,14,16],
            ]
        humansd_skeleton_width=10
        humansd_color=sns.color_palette("hls", len(humansd_skeleton)) 
        
        def plot_kpts(img_draw, kpts, color, edgs,width):     
                for idx, kpta, kptb in edgs:
                    if kpts[kpta,2]>mmpose_detection_thresh and \
                        kpts[kptb,2]>mmpose_detection_thresh :
                        line_color = tuple([int(255*color_i) for color_i in color[idx]])
                        
                        cv2.line(img_draw, (int(kpts[kpta,0]),int(kpts[kpta,1])), (int(kpts[kptb,0]),int(kpts[kptb,1])), line_color,width)
                        cv2.circle(img_draw, (int(kpts[kpta,0]),int(kpts[kpta,1])), width//2, line_color, -1)
                        cv2.circle(img_draw, (int(kpts[kptb,0]),int(kpts[kptb,1])), width//2, line_color, -1)
        
        
        pose_image = np.zeros_like(image)
        for person_i in range(len(present_pose)):
            if np.sum(present_pose[person_i]["keypoints"])>0:
                plot_kpts(pose_image, present_pose[person_i]["keypoints"],humansd_color,humansd_skeleton,humansd_skeleton_width)
        
        return pose_image

    def make_batch_sd(
        self,
        image,
        pose_image,
        txt,
        device,
        num_samples=1,):
        batch={
            "jpg":(torch.from_numpy(image).to(dtype=torch.float32) / 255 *2 - 1.0),
            "pose_img": (torch.from_numpy(pose_image).to(dtype=torch.float32) / 255 *2 - 1.0),
            "txt": num_samples * [txt],
        }
        
        batch["pose_img"] = rearrange(batch["pose_img"], 'h w c -> 1 c h w')
        batch["pose_img"] = repeat(batch["pose_img"].to(device=device),
                            "1 ... -> n ...", n=num_samples)
        
        batch["jpg"] = rearrange(batch["jpg"], 'h w c -> 1 c h w')
        batch["jpg"] = repeat(batch["jpg"].to(device=device),
                            "1 ... -> n ...", n=num_samples)
        return batch

    def paint_humansd(self,humansd_sampler, image, pose_image, prompt, t_enc, seed, scale, device, num_samples=1, callback=None,
          do_full_sample=False,negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"):
        model = humansd_sampler.model
        seed_everything(seed)

        with torch.no_grad():
            batch = self.make_batch_sd(
                image,pose_image, txt=prompt, device=device, num_samples=num_samples)
            z = model.get_first_stage_encoding(model.encode_first_stage(
                batch[model.first_stage_key]))  # move to latent space
            c = model.cond_stage_model.encode(batch["txt"])
            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck]
                if len(cc.shape) == 3:
                    cc = cc[..., None]
                cc = cc.to(memory_format=torch.contiguous_format).float()
                cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                c_cat.append(cc)
                
            c_cat = torch.cat(c_cat, dim=1)
            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, negative_prompt)
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
            if not do_full_sample:
                # encode (scaled latent)
                z_enc = humansd_sampler.stochastic_encode(
                    z, torch.tensor([t_enc] * num_samples).to(model.device))
            else:
                z_enc = torch.randn_like(z)
            # decode it
            samples = humansd_sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                    unconditional_conditioning=uc_full, callback=callback)
            x_samples_ddim = model.decode_first_stage(samples)
            result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
        return [Image.fromarray(img.astype(np.uint8)) for img in result]
    

@PREPROCESSORS.register_module('Pose-driven-image-generation', module_name='HumanSD-preprocessor')
class MyCustomPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.trainsforms = self.init_preprocessor(**kwargs)

    def __call__(self, results):
        return results

    # def init_preprocessor(self, **kwarg):
    #     """ Provide default implementation based on preprocess_cfg and user can reimplement it.
    #         if nothing to do, then return lambda x: x
    #     """
    #     return lambda x: x


@PIPELINES.register_module('Pose-driven-image-generation', module_name='HumanSD-pipeline')
class MyCustomPipeline(Pipeline):
    """ Give simple introduction to this pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline
    >>> input = "Hello, ModelScope!"
    >>> my_pipeline = pipeline('my-task', 'my-model-id')
    >>> result = my_pipeline(input)

    """

    def __init__(self, model, preprocessor=MyCustomPreprocessor(), **kwargs):
        """
        use `model` and `preprocessor` to create a custom pipeline for prediction
        Args:
            model: model id on modelscope hub.
            preprocessor: the class of method be init_preprocessor
        """
        super().__init__(model=MyCustomModel(model), preprocessor=preprocessor,auto_collate=False)

    def _sanitize_parameters(self, **pipeline_parameters):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method
        considered to be a normal classmethod with default implementation / output

        Default Returns:
            Dict[str, str]:  preprocess_params = {}
            Dict[str, str]:  forward_params = {}
            Dict[str, str]:  postprocess_params = pipeline_parameters
        """
        return {}, pipeline_parameters, {}

    def _check_input(self, inputs):
        pass

    def _check_output(self, outputs):
        pass

    def forward(self, inputs, **forward_params):
        """ Provide default implementation using self.model and user can reimplement it
        """
        results = self.model(inputs["image_dir"],inputs["prompt"],inputs["neg_prompt"],inputs["sample_steps"],inputs["seed"],inputs["guidance_scale"],inputs["num_samples"])
        return results

    def postprocess(self, inputs):
        """ If current pipeline support model reuse, common postprocess
            code should be write here.

        Args:
            inputs:  input data

        Return:
            dict of results:  a dict containing outputs of model, each
                output should have the standard output name.
        """
        return inputs


# Tips: usr_config_path is the temporary save configuration location， after upload modelscope hub, it is the model_id
usr_config_path = './'
config = Config({
    "framework": 'pytorch',
    "task": 'Pose-driven-image-generation',
    "model": {'type': 'HumanSD'},
    "pipeline": {"type": "HumanSD-pipeline"},
    "allow_remote": False
})
config.dump('configuration.json')

if __name__ == "__main__":
    from modelscope.models import Model
    from modelscope.pipelines import pipeline
    #model = Model.from_pretrained(usr_config_path)
    image_dir = "fff1daa9ef182b86387d802dba686426e7feb396_19692096.jpg"
    prompt = "a woman"
    neg_prompt = "lmonochrome, lowres, bad anatomy, worst quality, low quality"
    inference = pipeline('Pose-driven-image-generation', model = "./")
    inputs = {"image_dir":image_dir,"prompt":prompt,"neg_prompt": neg_prompt,"sample_steps":30,"seed":None,"guidance_scale":10.0,"num_samples":2 }
    output = inference(input=inputs)
    output_dir = 'test'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(len(output)):
        output[i].save(os.path.join(output_dir,str(i).zfill(4)+'.jpg'))
