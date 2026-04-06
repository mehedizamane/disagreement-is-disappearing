This is the official repository and codebase for the ICWSM 26 paper titled "Disagreement is Disappearing on U.S. Cable Debate Shows". 

All sample codes are added here for reproducibility. The flow is as follows:

1. We download episodes from podcasts or tv archive using the sample-episode-download.py file
2. Next, we diarize the episodes to get transcription using whisper. The corresponding sample code is the sample-diarization.py file
3. Then, after we get the transcription, we clean the transcripts to remove ads, empty speaker turns, and create (host, guest) pairs to make our dataset ready for labeling through an LLM. The corresponding code for this can be found in the sample-cleaning.ipynb notebook
4. Next comes the most important part of our pipline---the labeling of (host, guest) pairs as either agreement, disagreement, or neutrality. This code can be found in the sample-labeling.py file
5. Finally, when we are done with the labeling and have our dataset all labeled, we can go ahead and use the sample-analysis-graphs.ipynb notebook to generate the analysis graphs that are seen in the main paper

We are working on incorporating more robustness to our cleaning step and are actively working on this project to get more podcast shows in our analyses. Stay tuned! 

If you have any questions or concerns, please reach out to sm.mehedi.zaman@rutgers.edu 

Also, please cite our paper if you happen to use the codebase or utilize any parts of our paper :) 

@article{zaman2025disagreement,
  title={Disagreement is Disappearing on US Cable Debate Shows},
  author={Zaman, SM and Garimella, Kiran},
  journal={arXiv preprint arXiv:2511.15774},
  year={2025}
}
