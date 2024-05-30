import argparse
import cv2
import torch
from PIL import Image
from torchvision import transforms
import numpy as np



def main():
    parser = argparse.ArgumentParser()
    # input parameter
    parser.add_argument('--input_image_paths', nargs='+', type=str, default=None)

    # process setting parameter
    parser.add_argument(
            '--process_type', 
            choices=[
                'resize', 
                'concat', 
                'to_edge_image', 
                'image_classification',
                'object_detection',
                'no_process'], 
            type=str, 
            default=None)

    # resize parameter
    parser.add_argument('--resize_width', type=int, default=None)
    parser.add_argument('--resize_height', type=int, default=None)

    # concat parameter
    parser.add_argument('--concat_direction', type=str, choices=['vertical', 'horizontal'])

    # to edge image parameter
    parser.add_argument('--thresholds', type=int, nargs='+', default=None)

    # image classification parameter
    parser.add_argument('--classification_model_name', type=str, choices=['resnet18'], default='resnet18')

    # object detection parameter
    parser.add_argument('--detection_model_name', type=str, choices=['yolo5'], default='yolo5')

    # output parameter
    parser.add_argument('--save_image_path', type=str, default=None)
    args = parser.parse_args()

    image_data = cv2.imread(args.input_image_paths[0])
    if args.process_type=='resize':
        image_data = cv2.resize(image_data, (args.resize_width, args.resize_height))
    elif args.process_type=='concat':
        image_data_list = [cv2.imread(path) for path in args.input_image_paths]
        if args.concat_direction=='vertical':
            image_data = cv2.vconcat(image_data_list)
        elif args.concat_direction=='horizontal':
            image_data = cv2.hconcat(image_data_list)
    elif args.process_type=='to_edge_image':
        image_data = cv2.Canny(image_data, args.thresholds[0], args.thresholds[1])
    elif args.process_type=='image_classification':
        # process of input data
        image_data = Image.open(args.input_image_paths[0])
        preprocess = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])])
        image_data_tensor = preprocess(image_data)
        image_data_tensor_batch = image_data_tensor.unsqueeze(0) # increase batch dimension

        # process of model
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.IMAGENET1K_V1')
        model.eval()
        if torch.cuda.is_available():
            image_data_tensor_batch = image_data_tensor_batch.to('cuda')
            model.to('cuda')

        # process of classification
        with torch.no_grad():
            output = model(image_data_tensor_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # show top categories
        with open("imagenet_labels/imagenet_classes.txt", "r") as file:
            categories = [s.strip() for s in file.readlines()]

        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())

        image_data = np.asarray(image_data)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        text_location = (100, 100)
        text_color=(255, 255, 255)
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        cv2.putText(
                image_data, 
                '{}  {:.3f}'.format(
                    categories[top5_catid[0]], 
                    top5_prob[0]), 
                text_location, 
                font, 
                fontScale=2.0, 
                color=text_color,
                thickness=3)
    elif args.process_type=='object_detection':
        # process of model 
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        image_data = cv2.imread(args.input_image_paths[0])
        #image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

        # Inference
        results = model([image_data])
        #results_tensor = results.xyxy[0]
        #print("results_tensor: {}".format(results_tensor))
        #results_pandas = results.pandas().xyxy[0]
        #print(results_pandas)

        image_data = results.render()[0]
        #print(image_data)
        

    elif args.process_type=='no_process':
        pass
    
    cv2.imwrite(args.save_image_path, image_data)
    

if __name__ == "__main__":
    main()

