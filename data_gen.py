# Generate datasets, Hao Wu
# Ignoring punctuation in data for now

import collections
import random

# group prefixes by the kind of post-fixes they can have:
# "no_postfix", "all_postfix", "help_postfix", "is_postfix"
# NEXT STEP: what if there are adjectives in instructions, e.g. quickly
def generateTrainingSet():
	find_prefix = {"all_postfix": ["locate", "grab", "fetch", "bring", \
			"find", "get", "search for", "can you find", "can you bring", "can you fetch", "can you get", \
			"can you grab", "can you search for", "can you locate"], \
			"help_postfix": ["help me find", "help me locate", "help me get", \
			"where is", "give me", "how can i find", "how can i get", \
			"help me fetch", "help me grab"], \
			"is_postfix": ["tell me where", "can you help me find where", "show me where"]}

	# TODO: Here we used a new phrasing "could you", we can test to see if this "could you"
	# phrase can be understood well when it is joined with other types of instructions
	display_prefix = {"help_postfix": ["show me what is in", "show me what's in", "show me", "open me"], \
					"all_postfix": ["display", "can you display", "show", "open", "access", \
					"cat", "can you show", "can you open", "can you cat", "can you access", \
					"could you show", "could you open", "could you display", "could you cat", \
					"could you access"]}

	# go to some directory
	# TODO: one challenge is that some instructions could carry umbigous meaning
	# e.g. access, open, could mean go to a directory or list everything in that directory
	# How to deal with this ambiguity needs to be considered.
	# navigate_prefix is not joined with any post_fixes on purpose
	navigate_prefix = {"no_postfix": ["go to", "can you go to", "please go to", "enter", "can you enter",\
	"please enter", "go in", "can you go in", "please go in", "into", "go into", "can you go into",
	"please go into", "get in", "can you get in", "please get in", "visit", "can you visit", \
	"please visit"]}

	# return to parent directory
	# Support more complex logic like go up certain levels
	go_up_prefix = {"no_postfix": ["go to parent", "go back to parent", "exit this level", "leave this level", "move to parent"], \
				"value_postfix": ["go up", "go out", "navigate up", "move up", "step up"], \
				"composite_postfix": ["go", "navigate", "move", "step"]}

	#return to previous directory, "cd -"
	go_back_prefix = {"all_postfix": ["return to the place before", "go back", "go back to where i was", \
					"return to previous location", "revert", "go back to the previous location", \
					"return to where i was", "go to the directory before", \
					"return to the directory before", "I want to go back", "can you go back"\
					"can you return to where i was", "can you return to previous location",
					"can you take me back", "take me back", "take me back to where i was", "step back"]}

	file_names = ["a.txt", "b.txt", "code.py", "instructions.pdf", "submissions.py", \
				"graders.c", "graders.cpp", "webserver.js", "dummy.back", "hello.html", \
				"both-small.png", "submission.pyc", "main.css", "hw3", "weights", "somefolder", \
				"image.jpg", "homework.doc", "lalala.docx", "android.java", "picture.jpeg"]

	# find_prefix and display_prefix can share this find_post_fix
	find_postfix = {"help_postfix": ["please", "thanks", "now", ""], \
				"is_postfix": ["is", "sits", "exists"], \
				"all_postfix": ["please", "thanks", "now", "for me", ""]}

	go_up_postfix = {"no_postfix": [""], \
					 "value_postfix": ['level', 'directory'], \
					 "composite_postfix": ['level up', 'directory up', 'steps up']}

	# first do a clean up
	# then write data-label pairs
	ins = open("data/instructions.txt", 'w')
	com = open("data/commands.txt", 'w')

	# generate find instruction/command pairs
	count = 0
	for pre in find_prefix:
		for fn in file_names:
			for pre_v in find_prefix[pre]:
				for post_v in find_postfix[pre]:
					count += 1
					ins.write(pre_v + " " + fn + " " + post_v + "\n")
					com.write("find / -name \"" + fn + "\"" + "\n")
	print("Num find instructions: {}\n".format(count))

	#generate display instruction/command pairs
	count = 0
	for pre in display_prefix:
		for fn in file_names:
			for pre_v in display_prefix[pre]:
				for post_v in find_postfix[pre]:
					count += 1
					ins.write(pre_v + " " + fn + " " + post_v + "\n")
					# first check if the argument is a foler or a file, the ls or cat
					com.write("if [[ -d " + fn + " ]]; then ls " + fn + "; else cat " + fn + "; fi; \n")

	print("Num display instructions: {}\n".format(count))
	
	count = 0
	for pre in navigate_prefix:
		for fn in file_names:
			for pre_v in navigate_prefix[pre]:
				count += 1
				# no post fixes on purpose, want see how it learns
				ins.write(pre_v + " " + fn + "\n")
				com.write("cd " + fn + "\n")
	print("Num navigate instructions: {}\n".format(count))
	
	# generate 'cd ../../..' etc instruction/command pairs
	count = 0
	for pre in go_up_prefix:
		for pre_v in go_up_prefix[pre]:
			for post_v in go_up_postfix[pre]:
				if pre == "no_postfix":
					count += 1
					ins.write(pre_v + "\n")
					com.write("cd ..\n")
				else:
					for i in range(1, 11):
						count += 1
						ins.write(pre_v + " {}".format(i) + " " + post_v + "\n")
						com.write("cd " + "../"*i + "\n")
	print("Num go up instructions: {}\n".format(count))
	# generate 'cd -' instruction/command pairs
	count = 0
	for pre in go_back_prefix:
		for pre_v in go_back_prefix[pre]:
			for post_v in find_postfix[pre]:
				count += 1
				ins.write(pre_v + " " + post_v + "\n")
				com.write("cd -\n")
	print("Num go back instructions: {}\n".format(count))
	# TODO: add some other commands, EXTEND later

	# close file handles
	ins.close()
	com.close()

	# randomly shuffle data
	ins = open("data/instructions.txt", 'r')
	com = open("data/commands.txt", 'r')

	data = [ line for line in ins ]
	label = [ line for line in com ]

	c = list(zip(data, label))

	random.shuffle(c)

	data, label = zip(*c)

	with open('data/data.txt','w') as dataRand:
	    for line in data:
	        dataRand.write( line )

	with open('data/label.txt','w') as labelRand:
	    for line in label:
	        labelRand.write( line )

	ins.close()
	com.close()

	print("Finished generating Training data\n")

generateTrainingSet()
# I didn't want to just partition the data to training set and validation set
# I want the validation set to have some English instruction structures that
# does not exist in the training set at all.
def generateValidationSet():
	'''
	I want to see how the model deals with structures unseen in training!

	e.g. "bring me somefile" 
	structure didn't exist in the training dataset.
	
	e.g."could you fetch"
	structure also haven't been seen before

	I am not introducing any new vocabulary here because there is no point
	Again, some vocabulary was seen in other instructions
	but not this type of instruction, like the word could
	'''
	find_prefix = {"all_postfix": ["grab me", "fetch me", "bring me", \
			"get me", "could you fetch", "can you get"], \
			"help_postfix": ["can you help me find", "can you help me get"], \
			"is_postfix": ["can you tell me where", "can show me where"]}


	display_prefix = {"help_postfix": ["give me what's in", \
					"can you show me what is in", \
					"could you show what is in"]}

	navigate_prefix = {"no_postfix": ["can you please go to", "could you go in", \
					"please go into"]}

	
	# In the training set, there was no "please" or "can you" prefixes for this
	go_up_prefix = {"no_postfix": ["please go back to parent"], \
				"value_postfix": ["please go up", "can you go out"], \
				"composite_postfix": ["please go", "can you move"]}

	go_back_prefix = {"all_postfix": ["return to the location before", "can you go back", \
					"can you go back to where i was", "could you take me back"]}

	# Introducing new filenames, to see how the model learns what should be kept as argument
	# and not translate
	file_names = ["asdnoiwn.txt", "weirdstuff.py", "angular.js"]

	# find_prefix and display_prefix can share this find_post_fix
	find_postfix = {"help_postfix": ["please", "thanks", "now", ""], \
				"is_postfix": ["is", "sits", "exists"], \
				"all_postfix": ["please", "thanks", "now", "for me", ""]}

	go_up_postfix = {"no_postfix": [""], \
					 "value_postfix": ['level', 'directory'], \
					 "composite_postfix": ['level up', 'directory up', 'steps up']}


 	# first do a clean up
	# then write data-label pairs
	ins = open("data/validation/instructions.txt", 'w')
	com = open("data/validation/commands.txt", 'w')

	# generate find instruction/command pairs
	count = 0
	for pre in find_prefix:
		for fn in file_names:
			for pre_v in find_prefix[pre]:
				for post_v in find_postfix[pre]:
					count += 1
					ins.write(pre_v + " " + fn + " " + post_v + "\n")
					com.write("find / -name \"" + fn + "\"" + "\n")
	print("Num find instructions: {}\n".format(count))

	#generate display instruction/command pairs
	count = 0
	for pre in display_prefix:
		for fn in file_names:
			for pre_v in display_prefix[pre]:
				for post_v in find_postfix[pre]:
					count += 1
					ins.write(pre_v + " " + fn + " " + post_v + "\n")
					# first check if the argument is a foler or a file, the ls or cat
					com.write("if [[ -d " + fn + " ]]; then ls " + fn + "; else cat " + fn + "; fi; \n")

	print("Num display instructions: {}\n".format(count))
	
	count = 0
	for pre in navigate_prefix:
		for fn in file_names:
			for pre_v in navigate_prefix[pre]:
				count += 1
				# no post fixes on purpose, want see how it learns
				ins.write(pre_v + " " + fn + "\n")
				com.write("cd " + fn + "\n")
	print("Num navigate instructions: {}\n".format(count))
	
	# generate 'cd ../../..' etc instruction/command pairs
	count = 0
	for pre in go_up_prefix:
		for pre_v in go_up_prefix[pre]:
			for post_v in go_up_postfix[pre]:
				if pre == "no_postfix":
					count += 1
					ins.write(pre_v + "\n")
					com.write("cd ..\n")
				else:
					for i in range(1, 11):
						count += 1
						ins.write(pre_v + " {}".format(i) + " " + post_v + "\n")
						com.write("cd " + "../"*i + "\n")
	print("Num go up instructions: {}\n".format(count))
	# generate 'cd -' instruction/command pairs
	count = 0
	for pre in go_back_prefix:
		for pre_v in go_back_prefix[pre]:
			for post_v in find_postfix[pre]:
				count += 1
				ins.write(pre_v + " " + post_v + "\n")
				com.write("cd -\n")
	print("Num go back instructions: {}\n".format(count))
	# TODO: add some other commands, EXTEND later

	# close file handles
	ins.close()
	com.close()

	# randomly shuffle data
	ins = open("data/validation/instructions.txt", 'r')
	com = open("data/validation/commands.txt", 'r')

	data = [ line for line in ins ]
	label = [ line for line in com ]

	c = list(zip(data, label))

	random.shuffle(c)

	data, label = zip(*c)

	with open('data/validation/data.txt','w') as dataRand:
	    for line in data:
	        dataRand.write( line )

	with open('data/validation/label.txt','w') as labelRand:
	    for line in label:
	        labelRand.write( line )

	ins.close()
	com.close()

	print("Finished generating validation set\n")


generateValidationSet()