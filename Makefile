run_preprocessed:
	python -c 'from data_restoration.preprocessing import preprocessed_data; preprocessed_data()'

run_preprocessed_small:
	python -c 'from data_restoration.preprocessing import preprocessed_data_small; preprocessed_data_small()'

run_api:
	uvicorn api.fast:app --reload

reinstall_package:
	@pip uninstall -y data_restoration || :
	@pip install -e .
