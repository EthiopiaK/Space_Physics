# Space_Physics
These scripts will process JPG spectrogram files in a given directory and attempt to identify Electromagnetic Ion Cyclotron (EMIC) waves using Scikit-image and OpenCV. 
converter.py is a preprocessing file to convert raw spectrograms to the '.jpg' format required by blobextractor.py. 
blobextractor.py uses edge detection and region props to isolate an EMIC event and output its frequency and time ranges to a txt file. 
clickable.py uses an OpenCV window to allow users to click on event boundaries. It outputs the frequency range, time range, and median amplitude of the clicked EMIC event to a txt file
A sample image has been uploaded to the repository. 
Below is a visualization of the output from blobextractor.py. 
<img width="266" height="74" alt="original" src="https://github.com/user-attachments/assets/ac7ff495-fbdb-41e9-a591-98d051e4b3fc" />
<img width="640" height="480" alt="regions" src="https://github.com/user-attachments/assets/fcfb5f19-1397-4fea-9724-749f362cd3bf" />

Note that in this specific example, the automated script has misidentified high-frequency noise as an EMIC event (a false positive). 
Thus, a clickable workflow was created to ensure accuracy. 
