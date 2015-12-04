def write_to_file(filename, opt, centers, edges, n, g, known_g, class_counter, acc, time):
	try:
		filename = filename + '-' + str(n) + '-' + str(known_g) + '.out'
		f = open(filename, 'w')
		
		f.write('n = ' + str(n))
		f.write('\n') 
		f.write('g = ' + str(g))
		f.write('\n') 
		f.write('known_g = ' + str(known_g))
		f.write('\n') 
		
		f.write('opt = ' + str(opt))
		f.write('\n') 
		f.write('centers = ' + str(centers))
		f.write('\n') 
		f.write('edges = ' + str(edges))
		f.write('\n')
		f.write('class counter = ' + str(class_counter))
		f.write('\n')

		if known_g:
			f.write('acc = ' + str(acc))
			f.write('\n') 
		f.write('time = ' + str(time))
		f.write('\n')

		print filename, 'saved.'

	except ValueError:
		print "Error in output saving."