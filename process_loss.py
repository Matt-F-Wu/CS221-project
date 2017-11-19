loss_file = open("all_loss.dat", 'r')
train_loss = open("tr_loss.csv", 'w')
val_loss = open("va_loss.csv", 'w')

content = loss_file.readlines()

tr = True
for line in content:
	if tr:
		train_loss.write(line)
		tr = False
	else:
		val_loss.write(line)
		tr = True