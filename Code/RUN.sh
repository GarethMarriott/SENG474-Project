for YEAR in 2009 2010 2011 2012 2013 2014 2015 2016 2017
do
  for WEEK in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
  do
    python prediction_values.py $YEAR $WEEK "./data/"$YEAR"-"$WEEK".txt" 0
    sleep 1
  done
  python prediction_values.py $YEAR 16 "./point-results-"$YEAR".txt" 1
done

sleep 5
python perceptron.py
