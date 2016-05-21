## declare an array variable
declare -a arr=("reporter" "interpreter" "manager" "superior")

## now loop through the above array
for condition in "${arr[@]}"
do
	tail -n +2 comments.csv | cut -d ',' -f 4,5 | grep -i $condition > comments-${condition}
done
