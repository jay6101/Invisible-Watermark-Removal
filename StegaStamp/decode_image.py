import bchlib
import glob
import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    for filename in files_list:
        file_id = filename.split('/')[-1].split('.')[0].split('_')[0]
        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
        image /= 255.

        feed_dict = {input_image:[image]}

        secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]
        decoded_secret = ''.join([str(int(x)) for x in secret.tolist()])
        print(f'{filename}: {decoded_secret}')
        
        # Save decoded secret to file if save_dir is provided
        if args.save_dir is not None:
            save_name = filename.split('/')[-1].split('.')[0]
            output_file = os.path.join(args.save_dir, f'{save_name}_decoded.txt')
            with open(output_file, 'w') as f:
                f.write(decoded_secret)
            print(f'Saved to: {output_file}')


        # packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
        # packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        # packet = bytearray(packet)

        # data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

        # bitflips = bch.decode_inplace(data, ecc)

        # if bitflips != -1:
        #     try:
        #         code = data.decode("utf-8")
        #         print(filename, code)
        #         continue
        #     except:
        #         continue
        # print(filename, 'Failed to decode')


if __name__ == "__main__":
    main()
