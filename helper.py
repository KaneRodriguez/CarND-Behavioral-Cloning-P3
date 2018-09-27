'''
Helper for logging information on previous model training session
'''
import datetime
import csv

def update_log(history_object, logPath=r'log.csv', arch_title='Architecture', changes='""'):
    fields=[str(datetime.date.today()),
            str(datetime.datetime.now().strftime('%H:%M')),
            history_object.history["loss"][-1],
            history_object.history["val_loss"][-1], 
            architecture_title,  
            len(history_object.history["loss"]), # of Epochs 
            batch_size,
            notable_changes,
            '""']

    # append new log data to end of log file
    with open(logPath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        
'''
    Helper for plotting multiple images and giving them titles
'''
import numpy as np
import matplotlib.pyplot as plt
import math

def plotImages(images, titles=[""], columns=1, figsize=(20,10), gray=False, saveAs=''):
    errorStr = "plotImages failed..."
    # images and titles must be lists
    if(not isinstance(images, (list,)) or not isinstance(titles, (list,))):
        print(errorStr + " images/titles are not both instances of list")
        return
    
    # the number of titles must match the number of columns OR
    # match the number of images
    if(len(titles) != columns and len(titles) != len(images)):
        print(errorStr + " images/titles are not the same length")
        return
    
    plt.figure(figsize=figsize)
    
    fig = plt.gcf()
    
    for i, image in enumerate(images):
        rows = math.ceil(len(images) / columns)
        plt.subplot(rows, columns, i + 1)
        
        if len(images) == len(titles):
            plt.gca().set_title(titles[i])
        else:
            plt.gca().set_title(titles[i % columns])
       
        # if gray is a list, each item  
        # corresponds to if each row is gray
        tmpGray = gray
        if isinstance(gray, (list,)):
            tmpGray = gray[i // columns]
            
        if gray:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)

    if saveAs != '':
        fig.savefig(saveAs, dpi=fig.dpi)