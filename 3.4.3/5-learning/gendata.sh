# run 100 instances with max 20 moves with 1 ghost and non-moving pacman both randomally positioned to start
python gendata.py RandGhost_easy -n 100 -q -i 20 -k 1 -g RandomGhost -p StopAgent -l freespace_20x20,freespace_50x50 --randPos
python gendata.py SeekerGhost_easy -n 100 -q -i 20 -k 1 -g DirectionalGhost -p StopAgent -l freespace_20x20,freespace_50x50 --randPos
python gendata.py TrackerGhost_easy -n 100 -q -i 20 -k 1 -g TrackingGhost -p StopAgent -l freespace_20x20,freespace_50x50 --randPos

# as above but run over *all* levels
python gendata.py RandomGhost -n 100 -q -i 20 -k 1 -g RandomGhost -p StopAgent --randPos
python gendata.py SeekerGhost -n 100 -q -i 20 -k 1 -g DirectionalGhost -p StopAgent --randPos
python gendata.py TrackerGhost -n 100 -q -i 20 -k 1 -g TrackingGhost -p StopAgent --randPos


