.PHONY: repl
repl:
	@mkdir -p out
	julia --project=. --threads=auto -i main.jl