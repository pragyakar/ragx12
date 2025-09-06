# Top-level Makefile to proxy into ragX12-api service

.PHONY: api api-install

api-install:
	$(MAKE) -C ragX12-api install

api:
	$(MAKE) -C ragX12-api ragx12-api

tunnel-api: 
	$(MAKE) -C tunnel-ragX12-api tunnel-ragx12-api	
