#!/usr/bin/python

# Script to dump TensorFlow weights in TRT v1 and v2 dump format.
# The V1 format is for TensorRT 4.0. The V2 format is for TensorRT 4.0 and later.

import sys
import struct
import argparse
try:
    import tensorflow as tf
    from tensorflow.python import pywrap_tensorflow
except ImportError as err:
    sys.stderr.write("""Error: Failed to import module ({})""".format(err))
    sys.exit()

parser = argparse.ArgumentParser(description='TensorFlow Weight Dumper')

parser.add_argument('-m', '--model', required=True, help='The checkpoint file basename, example basename(model.ckpt-766908.data-00000-of-00001) -> model.ckpt-766908')
parser.add_argument('-o', '--output', required=True, help='The weight file to dump all the weights to.')
parser.add_argument('-1', '--wtsv1', required=False, default=False, type=bool, help='Dump the weights in the wts v1.')

opt = parser.parse_args()

if opt.wtsv1:
    print "Outputting the trained weights in TensorRT's wts v1 format. This format is documented as:"
    print "Line 0: <number of buffers in the file>"
    print "Line 1-Num: [buffer name] [buffer type] [buffer size] <hex values>"
else:
    print "Outputting the trained weights in TensorRT's wts v2 format. This format is documented as:"
    print "Line 0: <number of buffers in the file>"
    print "Line 1-Num: [buffer name] [buffer type] [(buffer shape{e.g. (1, 2, 3)}] <buffer shaped size bytes of data>"

inputbase = opt.model
outputbase = opt.output

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def getTRTType(tensor):
    if tf.as_dtype(tensor.dtype) == tf.float32:
        return 0
    if tf.as_dtype(tensor.dtype) == tf.float16:
        return 1
    if tf.as_dtype(tensor.dtype) == tf.int8:
        return 2
    if tf.as_dtype(tensor.dtype) == tf.int32:
        return 3
    print(f"Tensor data type of {tensor.dtype} is not supported in TensorRT")
    sys.exit();

try:
   # Open output file
    outputFileName = f"{outputbase}.wts" if opt.wtsv1 else f"{outputbase}.wts2"
    with open(outputFileName, 'w') as outputFile:
        # read vars from checkpoint
        reader = pywrap_tensorflow.NewCheckpointReader(inputbase)
        var_to_shape_map = reader.get_variable_to_shape_map()

        count = sum(1 for _ in sorted(var_to_shape_map))
        outputFile.write("%s\n"%(count))

            # Dump the weights in either v1 or v2 format
        for key in sorted(var_to_shape_map):
            tensor = reader.get_tensor(key)
            file_key = key.replace('/','_')
            typeOfElem = getTRTType(tensor)
            val = tensor.shape
            if opt.wtsv1:
                val = tensor.size
            print(f"{file_key} {typeOfElem} {val} ")
            flat_tensor = tensor.flatten()
            outputFile.write(f"{file_key} 0 {val} ")
            if opt.wtsv1:
                for weight in flat_tensor:
                    hexval = float_to_hex(float(weight))
                    outputFile.write(f"{hexval[2:]} ")
            else:
                outputFile.write(flat_tensor.tobytes())
            outputFile.write("\n");
except Exception as e:  # pylint: disable=broad-except
    print(e)
    if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
                "with SNAPPY.")
        if "Data loss" in str(e) and any(
            e in inputbase for e in [".index", ".meta", ".data"]
        ):
            proposed_file = ".".join(inputbase.split(".")[:-1])
            v2_file_error_template = """
           It's likely that this is a V2 checkpoint and you need to provide the filename
           *prefix*.  Try removing the '.' and extension.  Try:
           inspect checkpoint --file_name = {}"""
            print(v2_file_error_template.format(proposed_file))
