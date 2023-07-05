# Characterization

This folder contains the characterization files we used to investigate the inference dataset.

## Frequent words vs tactics

This notebook confirms that it is the context a bash word "lives" in that determine its prediction (i.e., context matters). This justify the use of attention-based model such as LMs.

## Sessions vs Fingerprints

If we consider the sequences of words' predictions, we can define **fingerprints**. Those fingerprints capture the attackers' targets when attacking a machine, and group together different sessions having the same objectives. This notebook studies how many sessions are assigned per fingerprint and the fingerprint's share (how many sessions share the same fingerprint).

## Forensic - Peak Detection - Dota - Dhpcd - Clustering

Examples of how one could leverage the fingerprints to automatize the experts' job while solving tasks such as Forensic analysis, peak detection and sessions' morphing.
