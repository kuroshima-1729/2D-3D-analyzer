import argparse
import cv2


def main():
    parser = argparse.ArgumentParser()
    # input parameter
    parser.add_argument('--input_image_paths', nargs='+', type=str, default=None)

    # process setting parameter
    parser.add_argument('--process_type', choices=['resize', 'concat', 'to_edge_image', 'no_process'], type=str, default=None)

    # resize parameter
    parser.add_argument('--resize_width', type=int, default=None)
    parser.add_argument('--resize_height', type=int, default=None)

    # concat parameter
    parser.add_argument('--concat_direction', type=str, choices=['vertical', 'horizontal'])

    # to edge image parameter
    parser.add_argument('--thresholds', type=int, nargs='+', default=None)

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
    elif args.process_type=='no_process':
        pass

    cv2.imwrite(args.save_image_path, image_data)
    

if __name__ == "__main__":
    main()

