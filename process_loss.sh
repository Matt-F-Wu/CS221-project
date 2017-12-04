mv results/all_loss.dat results/all_loss.dat.rough
grep -Eo "perplexity [0-9\.]+" ./results/all_loss.dat | sed "s/perplexity //g" > ./results/all_loss.dat
python process_loss.py
