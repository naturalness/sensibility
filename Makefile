FAST_DIR = /dev/shm

# 1,578,048,815 * 5% / 10 = 7,890,244
TOKENS_PER_FOLD = 7890244

VECTORS = javascript.sqlite3
ASSIGNED_VECTORS = $(FAST_DIR)/$(VECTORS)

$(ASSIGNED_VECTORS): $(VECTORS)
	cp $(VECTORS) $(ASSIGNED_VECTORS)
	chmod u+w $(ASSIGNED_VECTORS)
	./place_into_folds.py --overwrite --folds 10 --min-tokens $(TOKENS_PER_FOLD) $(ASSIGNED_VECTORS) \
		|| (rm -f $(ASSIGNED_VECTORS) && false)
