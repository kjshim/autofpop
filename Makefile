all : setup.py myauth.py autodora.py andlib.py authserver.py
	python setup.py py2exe
	rm -rf dist\tcl
	upx -9 dist\Autodora.exe
	rm dist\authserver.exe
	mv dist autodora_dist
	7z a autodora_dist.zip autodora_dist\*


clean :
	rm -rf dist
	rm -rf build
	rm -rf autodora_dist
	rm -rf autodora_dist.zip
	rm -rf *.pyc
	rm img.raw
	rm my.sh

test:
	nosetests
