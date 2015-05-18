workon documentparserator
mkdir xml
aws s3 cp --recursive s3://lensnola/contracts/contract_amounts/human_labels/ xml --acl public-read
parserator train "./xml/*.xml" contract_parser
deactivate