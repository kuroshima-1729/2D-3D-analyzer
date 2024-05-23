import argparse
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_paths', nargs='+', type=str, default=None)
    parser.add_argument('--process_type', choices=['resize', 'no_process'], type=str, default=None)

    # resize parameter
    parser.add_argument('--resize_width', type=int, default=None)
    parser.add_argument('--resize_height', type=int, default=None)

    parser.add_argument('--save_image_path', type=str, default=None)
    args = parser.parse_args()

    image_data = cv2.imread(args.input_image_paths[0])
    if args.process_type=='resize':
        image_data = cv2.resize(image_data, (args.resize_width, args.resize_height))
    elif args.process_type=='no_process':
        pass

    cv2.imwrite(args.save_image_path, image_data)
    

if __name__ == "__main__":
    main()

