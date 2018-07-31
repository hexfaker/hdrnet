.PHONY: clean-image clean-old-images image

all: image

image:
	docker build -t hdrnet:latest .

clean-old-images:
	bash -c "docker rmi $(docker images | grep '^<none>' | awk '{print $3}')"

clean-image:
	docker rmi hdrnet:latest
