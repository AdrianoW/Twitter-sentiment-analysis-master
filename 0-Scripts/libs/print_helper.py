import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Ex:
    # Compute confusion matrix
	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
	                      title='Confusion matrix, without normalization')

	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
	                      title='Normalized confusion matrix')

	plt.show()
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """
    pretty print for confusion matrixes

    Ex:
	    # first generate with specified labels
		labels = [ ... ]
		cm = confusion_matrix(ypred, y, labels)

		# then print it in a pretty way
		print_cm(cm, labels)

    """
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print

#def old():
#     import subprocess

#     IMG_FOLDER = '/Users/adrianow/Documents/Google Drive/Mestrado/Tese/programacao/Tese/0-Scripts/img/'
#     filename = IMG_FOLDER+'out.html'
#     outname = IMG_FOLDER+'out.png'
#     cropname = IMG_FOLDER+'cropped.png'

#     with open(filename, 'wb') as f:
#         t = pipe.pprint_results(pd.concat([res_df, bool_res]))
#         t.set_table_styles([{'selector': 'td', 'props': [('border', '1px solid gray')]},\
#                             {'selector': '', 'props': [('border-collapse', 'collapse'),
#                                                        ('border-spacing', 0),
#                                                        ('background-color','transparent'),
#                                                        ('display', 'table'),
#                                                        ('border-color', 'grey'),
#                                                        ('font-family', '"Helvetica Neue", Helvetica, Arial, sans-serif')]},\
#                             {'selector': 'th', 'props': [('border', '1px solid gray')]}])
#         f.write(t.render())
#     rasterize =  '/Users/adrianow/Documents/Programs/phantomjs-2.1.1-macosx/examples/rasterize.js'
#     subprocess.call(['/Users/adrianow/Documents/Programs/phantomjs-2.1.1-macosx/bin/phantomjs', rasterize, filename, outname])
#     subprocess.call(['convert', outname, '-trim', cropname])