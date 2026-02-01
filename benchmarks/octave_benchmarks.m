% SPGL1 Octave/MATLAB Benchmarks
%
% This script benchmarks MATLAB/Octave SPGL1 functions to compare with Python.
% Run with: octave --quiet octave_benchmarks.m [output_file] [sizes]
%
% Example:
%   octave --quiet octave_benchmarks.m results/micro_octave.csv small,medium,large
%
% The output CSV format matches the Python benchmark output for easy comparison.

function octave_benchmarks(output_file, sizes_str)
    % Add MATLAB SPGL1 to path
    matlab_spgl_path = fullfile(getenv('HOME'), 'code', 'matlab_spgl');
    addpath(matlab_spgl_path);
    addpath(fullfile(matlab_spgl_path, 'private'));

    % Default arguments
    if nargin < 1 || isempty(output_file)
        output_file = 'results/micro_octave.csv';
    end
    if nargin < 2 || isempty(sizes_str)
        sizes_str = 'small,medium,large';
    end

    % Parse sizes
    sizes = strsplit(sizes_str, ',');

    % Configuration
    num_warmup = 3;
    num_runs = 10;

    % Problem size definitions
    size_configs = struct();
    size_configs.tiny = struct('n', 100, 'm', 50, 'sparsity', 0.1);
    size_configs.small = struct('n', 1000, 'm', 500, 'sparsity', 0.05);
    size_configs.medium = struct('n', 10000, 'm', 5000, 'sparsity', 0.05);
    size_configs.large = struct('n', 100000, 'm', 50000, 'sparsity', 0.01);
    size_configs.very_large = struct('n', 1000000, 'm', 100000, 'sparsity', 0.001);

    % Open output file
    fid = fopen(output_file, 'w');
    if fid == -1
        error('Could not open output file: %s', output_file);
    end

    % Write CSV header
    fprintf(fid, 'function_name,size_category,n,median_time_ms,std_time_ms,min_time_ms,max_time_ms,num_runs\n');

    % Run benchmarks for each size
    for s = 1:length(sizes)
        size_name = strtrim(sizes{s});
        if ~isfield(size_configs, size_name)
            fprintf('Warning: Unknown size "%s", skipping\n', size_name);
            continue;
        end

        cfg = size_configs.(size_name);
        fprintf('\n%s\n', repmat('=', 1, 60));
        fprintf('Problem size: %s (n=%d)\n', size_name, cfg.n);
        fprintf('%s\n', repmat('=', 1, 60));

        % Generate test data
        data = generate_test_data(cfg);

        % Run each benchmark
        benchmarks = get_benchmarks();
        for b = 1:length(benchmarks)
            bench = benchmarks{b};
            try
                result = run_benchmark(bench, data, num_warmup, num_runs);
                result.size_category = size_name;
                result.n = cfg.n;

                fprintf('  %-30s: %10.4f ms (+/- %.4f)\n', ...
                    result.function_name, result.median_time_ms, result.std_time_ms);

                % Write to CSV
                fprintf(fid, '%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%d\n', ...
                    result.function_name, result.size_category, result.n, ...
                    result.median_time_ms, result.std_time_ms, ...
                    result.min_time_ms, result.max_time_ms, result.num_runs);
            catch ME
                fprintf('  %-30s: ERROR - %s\n', bench.name, ME.message);
            end
        end
    end

    fclose(fid);
    fprintf('\nResults saved to %s\n', output_file);
end


function data = generate_test_data(cfg)
    % Generate test data for benchmarks
    %
    % Parameters:
    %   cfg : struct with fields n, m, sparsity
    %
    % Returns:
    %   data : struct with test vectors and matrices

    rng(42);  % Set seed for reproducibility
    n = cfg.n;
    m = cfg.m;

    % Basic vectors
    data.n = n;
    data.m = m;
    data.x = randn(n, 1);
    data.b = randn(n, 1);
    data.d = abs(randn(n, 1)) + 0.1;  % Positive weights

    % Sparse signal
    data.x_sparse = zeros(n, 1);
    nnz = round(n * cfg.sparsity);
    idx = randperm(n, nnz);
    data.x_sparse(idx) = randn(nnz, 1);

    % Tau values
    data.tau = norm(data.x_sparse, 1) * 0.5;

    % For L-BFGS
    data.k = 8;
    data.g1 = randn(n, 1);
    data.g2 = data.g1 + 0.1 * randn(n, 1);
    data.p = -data.g1;
    data.step = 0.1;

    % For productB - precompute sqrt vectors
    i = (1:n)';
    data.sqrt1 = sqrt(1 ./ (i .* (i + 1)));
    data.sqrt2 = sqrt(i ./ (i + 1));

    % For group norms
    data.g_groups = min(10, floor(n / 10));

    % For findLambdaStar
    data.z = abs(randn(n, 1));
    data.w = ones(n, 1);
    data.mu = 0.1;

    data.weights = 1.0;
    data.weights_vec = data.d;
end


function benchmarks = get_benchmarks()
    % Return cell array of benchmark definitions
    % Each benchmark is a struct with:
    %   name : string
    %   func : function handle that takes data struct and returns nothing

    benchmarks = {
        % Projection functions (Priority 1)
        struct('name', 'oneprojector', ...
               'func', @(d) oneProjector(d.b, 1.0, d.tau))

        struct('name', 'oneprojector_weighted', ...
               'func', @(d) oneProjector(d.b, d.d, d.tau))

        struct('name', 'oneprojector_i', ...
               'func', @(d) oneProjectorMex(abs(d.b), d.tau))

        struct('name', 'oneprojector_d', ...
               'func', @(d) oneProjectorMex(abs(d.b), d.d, d.tau))

        % Norm functions (Priority 1)
        struct('name', 'norm_l1_primal', ...
               'func', @(d) NormL1_primal(d.weights, d.x))

        struct('name', 'norm_l1_primal_weighted', ...
               'func', @(d) NormL1_primal(d.weights_vec, d.x))

        struct('name', 'norm_l1_dual', ...
               'func', @(d) NormL1_dual(d.weights, d.x))

        struct('name', 'norm_l1_project', ...
               'func', @(d) NormL1_project(d.x, d.weights, d.tau))

        % Group norms (Priority 2)
        struct('name', 'norm_l12_primal', ...
               'func', @(d) run_norm_l12_primal(d))

        struct('name', 'norm_l12_dual', ...
               'func', @(d) run_norm_l12_dual(d))

        struct('name', 'norm_l12_project', ...
               'func', @(d) run_norm_l12_project(d))

        % Dual objective (Priority 2)
        struct('name', 'find_lambda_star', ...
               'func', @(d) findLambdaStar(d.z, d.w, d.tau, d.mu))

        % L-BFGS (Priority 2)
        struct('name', 'lbfgs_init', ...
               'func', @(d) lbfgsinit(d.n, d.k, 1.0))

        struct('name', 'lbfgs_hprod', ...
               'func', @(d) run_lbfgs_hprod(d))

        struct('name', 'lbfgs_bprod', ...
               'func', @(d) run_lbfgs_bprod(d))

        struct('name', 'lbfgs_update', ...
               'func', @(d) run_lbfgs_update(d))

        % ProductB (baseline - MEX compiled)
        struct('name', 'product_b_forward', ...
               'func', @(d) productBMex(d.x, 0, d.sqrt1, d.sqrt2))

        struct('name', 'product_b_transpose', ...
               'func', @(d) productBMex([0; d.x], 1, d.sqrt1, d.sqrt2))

        struct('name', 'compute_sqrt_vectors', ...
               'func', @(d) compute_sqrt_vecs(d.n))
    };
end


% Helper functions to prepare arguments for benchmarks

function result = run_norm_l12_primal(d)
    g = d.g_groups;
    n_adj = floor(d.n / g) * g;
    x = d.x(1:n_adj);
    num_groups = n_adj / g;
    weights = ones(num_groups, 1);
    result = NormL12_primal(g, x, weights);
end

function result = run_norm_l12_dual(d)
    g = d.g_groups;
    n_adj = floor(d.n / g) * g;
    x = d.x(1:n_adj);
    num_groups = n_adj / g;
    weights = ones(num_groups, 1);
    result = NormL12_dual(g, x, weights);
end

function result = run_norm_l12_project(d)
    g = d.g_groups;
    n_adj = floor(d.n / g) * g;
    x = d.x(1:n_adj);
    num_groups = n_adj / g;
    weights = ones(num_groups, 1);
    tau = norm(x) * 0.5;
    result = NormL12_project(g, x, weights, tau);
end

function result = run_lbfgs_hprod(d)
    % Initialize and add some history
    H = lbfgsinit(d.n, d.k, 1.0);
    for i = 1:min(3, d.k)
        g1 = d.g1 + (i-1) * 0.01 * d.x;
        g2 = g1 + 0.1 * d.x;
        p = -g1;
        [H, ~] = lbfgsupdate(H, d.step, p, g1, g2);
    end
    result = lbfgshprod(H, d.g1);
end

function result = run_lbfgs_bprod(d)
    H = lbfgsinit(d.n, d.k, 1.0);
    for i = 1:min(3, d.k)
        g1 = d.g1 + (i-1) * 0.01 * d.x;
        g2 = g1 + 0.1 * d.x;
        p = -g1;
        [H, ~] = lbfgsupdate(H, d.step, p, g1, g2);
    end
    result = lbfgsbprod(H, d.g1);
end

function result = run_lbfgs_update(d)
    H = lbfgsinit(d.n, d.k, 1.0);
    [H, noup] = lbfgsupdate(H, d.step, d.p, d.g1, d.g2);
    result = H;
end

function [sqrt1, sqrt2] = compute_sqrt_vecs(n)
    i = (1:n)';
    sqrt1 = sqrt(1 ./ (i .* (i + 1)));
    sqrt2 = sqrt(i ./ (i + 1));
end


function result = run_benchmark(bench, data, num_warmup, num_runs)
    % Run a single benchmark with warmup and timing
    %
    % Parameters:
    %   bench : struct with name and func
    %   data : test data struct
    %   num_warmup : number of warmup runs
    %   num_runs : number of timed runs
    %
    % Returns:
    %   result : struct with timing statistics

    % Warmup runs
    for i = 1:num_warmup
        bench.func(data);
    end

    % Timed runs
    times = zeros(num_runs, 1);
    for i = 1:num_runs
        tic;
        bench.func(data);
        times(i) = toc * 1000;  % Convert to ms
    end

    result.function_name = bench.name;
    result.median_time_ms = median(times);
    result.std_time_ms = std(times);
    result.min_time_ms = min(times);
    result.max_time_ms = max(times);
    result.num_runs = num_runs;
end


% Run if called as script
if ~isdeployed
    args = argv();
    if length(args) >= 2
        octave_benchmarks(args{1}, args{2});
    elseif length(args) >= 1
        octave_benchmarks(args{1}, 'small,medium,large');
    else
        octave_benchmarks('results/micro_octave.csv', 'small,medium,large');
    end
end
