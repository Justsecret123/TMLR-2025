#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


import os 

# Path to the subtitles folders
PATH = "../../../MovieGraphs_Data/Subtitles/clip_srt/"
directories = os.listdir("../../../MovieGraphs_Data/Subtitles/clip_srt/")


def main():
    """
    Main : converts MovieGraph subtitle files
    from WINDOWS-1252 to UTF-8
    """

    # Loop through the directories 
    for directory in directories:
        # Get the current directory files 
        files = os.listdir(PATH + f"{directory}/")
        # Loop through the files 
        for file in files:
            # Get the file name without extension
            filename = os.path.splitext(file)[0]
            # Try to convert the file 
            try: 
                # Execute the command 
                os.system(
                    f"iconv -f WINDOWS-1252 -t UTF-8 {PATH+directory+'/'+file} > {PATH+directory+'/'+filename}_utf8.webvtt")
                # Remove the old file
                os.system(f"rm {PATH+directory+'/'+file}")
            except:
                print(f"Could not convert {file}")
    

if __name__ == "__main__":
    main()
