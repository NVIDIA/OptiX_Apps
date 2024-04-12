// let's try to approximate the following integral from 0 to 10
// |sin(x)*x^2| * 1/sqrt(x!)

const g = 7;
const C = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];

const gamma = (z) => {
    if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
    else {
        z -= 1;

        var x = C[0];
        for (var i = 1; i < g + 2; i++)
        x += C[i] / (z + i);

        var t = z + g + 0.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, (z + 0.5)) * Math.exp(-t) * x;
    }
}

const f = (x) => {
    return Math.abs(Math.sin(x) * x*x) / Math.sqrt(Math.abs(x * x * x));
}

const f2 = (x) => {
    return Math.abs(x*x) / Math.sqrt(Math.abs(x * x * x));
}

const f3 = (x) => {
    // return 1 / Math.sqrt(Math.abs(x * x * x));
    return 0.1;
}

const uniform_sample = (start, stop) => {
    return Math.random() * (stop - start) + start;
}

const range = [0, 10];
const num_samples = 1000;
const num_candidates = 64;

const riemann_naive = (num_samples) => {
    let sum = 0;
    let step = (range[1] - range[0]) / num_samples;
    for(let i = range[0] + step; i < range[1]; i += step){
        sum += step * f(i);
        // console.log(step * f(i))
        if(isNaN(step * f(i))) console.log(i)
    }
    return sum;
}

const monte_carlo_naive = (num_samples) => {
    const volume = range[1] - range[0];
    let sum = 0;
    for(let i = 0; i < num_samples; i++){
        sum += f(uniform_sample(...range));
    }
    return volume * sum / num_samples;
}

const monte_carlo_importance_sampling = (num_samples, pdf) => {

}

const resampled_importance_sampling_1 = (target_pdf, f) => {
    const volume = range[1] - range[0];
    let samples = [];
    let weights = [];
    let w_sum = 0;
    for(let i = 0; i < num_candidates; i++){
        let sample = uniform_sample(...range);
        samples.push(sample);
        
        let w = f(sample) / (1 / 100);
        weights.push(w);
        w_sum += w;
    }

    let y = samples[Math.random() * num_candidates | 0];
    return f(y) / target_pdf(y) * (1 / num_candidates) * (weights.reduce((a, b) => a + b)) / volume;
}


const resampled_importance_sampling_N = (num_samples, target_pdf, f) => {
    const volume = range[1] - range[0];
    let sum = 0;
    
    for(let i = 0; i < num_samples; i++){
        let samples = [];
        let weights = [];
        let w_sum = 0;

        for(let i = 0; i < num_candidates; i++){
            let sample = uniform_sample(...range);
            samples.push(sample);
            
            let w = target_pdf(sample) / (1 / 100);
            weights.push(w);
            w_sum += w;
        }
    
        let y = samples[Math.random() * num_candidates | 0];
        sum += f(y) / target_pdf(y) * (1 / num_candidates) * (weights.reduce((a, b) => a + b));
    }

    return sum / num_samples / volume;
}

console.log(`Naive Riemann: ${riemann_naive(num_samples)}`)

console.log(`Naive MC: ${monte_carlo_naive(num_samples)}`)

// console.log(`Importance MC: ${monte_carlo_importance_sampling(num_samples, adjusted_sample_b)}`)

console.log(`RIS 1-estimator MC: ${resampled_importance_sampling_1(f2, f)}`)

console.log(`RIS N-estimator MC w/ f2: ${resampled_importance_sampling_N(num_samples, f2, f)}`)

console.log(`RIS N-estimator MC w/ f3: ${resampled_importance_sampling_N(num_samples, f3, f)}`)

console.log(`RIS N-estimator MC w/ f2: ${resampled_importance_sampling_N(num_samples * 100, f2, f)}`)

console.log(`RIS N-estimator MC w/ f3: ${resampled_importance_sampling_N(num_samples * 100, f3, f)}`)

