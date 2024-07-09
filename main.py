import gradio as gr
import cv2
import torch
import torch.nn as nn
import os
from utils.model import select_model
from utils.options import parse_args_function
from demo import demo

model_info = None

def play_video(video_file):
    try:
        assert model_info != None
    except:
        raise AssertionError("No model!")
    try:
        test_fourcc = cv2.VideoWriter_fourcc(*'avc1')
        test_writer = cv2.VideoWriter('./', test_fourcc, 60, (100, 100))
    except cv2.error:
        print("Create VideoWriter failed!")
        print("You may need to add \'OpenH264\' library to your current enviroment directory")
        print("You can download library from https://github.com/cisco/openh264/releases")
        raise RuntimeError

    dir = os.listdir()
    if('output' not in dir):
        os.mkdir('output')
    output_file = './output/output.mp4'
    if('temp_output' not in dir):
        os.mkdir('temp_output')
    temp_dir = './temp_output/'
    
    demo(model_info[0], model_info[1], model_info[2], \
         input_path = video_file, temp_path = temp_dir, output_path = output_file,\
         fourcc = cv2.VideoWriter_fourcc(*'avc1'))
    
    return output_file

# 创建 Gradio 接口
interface = gr.Interface(
    fn=play_video,
    inputs=gr.Video(),
    outputs=gr.Video(),
    title="GRASP Demo",
    description="上传一个 MP4 视频文件并进行播放"
)

# 启动接口
if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    args = parse_args_function()
    device_ids=args.gpu_number

    use_cuda = False
    if args.gpu:
        use_cuda = True

    model = select_model(args.model_def)

    if use_cuda and torch.cuda.is_available():
        print('Using GPU')
        # 模型加载到设备0
        model = model.cuda(device_ids[0])
        # 指定要用到的设备
        model = nn.DataParallel(model, device_ids=device_ids)

    if args.pretrained_model != '':
        model.load_state_dict(torch.load(args.pretrained_model))
    elif args.resume != '':
        print("resume from:"+args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
    else:
        pass

    model_info = (model, use_cuda, device_ids[0])

    interface.launch()