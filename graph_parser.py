from __future__ import print_function
from vgg16 import VGG16

class Graph(object):
	def __init__(self, program):
		self.program = program
		self.main_block = self.program.block(0)
		self.ops = self.main_block.ops
		self.op_size = len(self.ops)

	def print_op_details(self, index):
		if index < self.op_size:
			op = self.ops[index]

			print("-"*30 +"operator info" + "-"*30)
			print(str(op))

			print(op.type)
			print(op.input_arg_names)
			print(op.output_arg_names)
			print(op.attr_names)

			print("-"*30 +"input info" + "-"*30)
			for input_name in op.input_arg_names:
				input_var = self.main_block.var(input_name)
				print(str(input_var))

				print(input_var.dtype)
				print(input_var.shape)
				print("\n")

			print("-"*30 +"output info" + "-"*30)
			for output_name in op.output_arg_names:
				output_var = self.main_block.var(output_name)
				print(str(output_var))

				print(output_var.dtype)
				print(output_var.shape)
				print("\n")
		else:
			print("index is overflow!!!")



if __name__ == "__main__":
    vgg16 = VGG16(data_set="cifar10")
    vgg16.construct_vgg16_net(learning_rate=1e-3)

    inference_graph = Graph(vgg16.inference_program)
    train_graph = Graph(vgg16.train_program)
    
    print(inference_graph.op_size)
    print(train_graph.op_size)

    for op in inference_graph.ops:
    	print(op.type)

    print("-"*60)
    for op in train_graph.ops:
    	print(op.type)
    
    print("-"*60)
    train_graph.print_op_details(10)