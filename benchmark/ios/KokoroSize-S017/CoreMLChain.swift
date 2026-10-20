import CoreML
import Foundation

enum CoreMLChainError: Error, CustomStringConvertible {
    case missingModel(String)
    case missingOutput(String)
    case shape(String)

    var description: String {
        switch self {
        case .missingModel(let name): return "Missing Core ML model \(name)."
        case .missingOutput(let name): return "Missing Core ML output \(name)."
        case .shape(let message): return "Core ML shape error: \(message)"
        }
    }
}

// S017 chain: S014 student -> durations -> Kokoro int8 pitch -> sine source
// -> S015 v4-prune int8 vocoder -> inverse STFT. Every package takes any
// token count (1-512) and frame count (1-2048), so one loaded set serves
// every input. Alignment, the sine source, and inverse STFT are Swift.
final class CoreMLChain {
    static let maxTokens = 512
    static let maxFrames = 2048

    private let student: MLModel
    private let pitchEnergy: MLModel
    private let vocoder: MLModel

    init() throws {
        student = try Self.load("kokoro_student_w256")
        pitchEnergy = try Self.load("kokoro_pitch_int8_flex")
        // On the iPhone 12, .cpuAndGPU gets the app killed (no crash report) at
        // 567+ frames. CPU-only runs every length and matches the Mac gate run.
        vocoder = try Self.load("kokoro_s015_vocoder_int8", units: .cpuOnly)
    }

    // ids include the pad 0 at both ends. style is one af_heart row [256].
    func synthesize(ids inputIds: [Int64], style styleRow: [Float]) throws -> [Float] {
        let tokenCount = inputIds.count
        guard (1...Self.maxTokens).contains(tokenCount), styleRow.count == 256 else {
            throw CoreMLChainError.shape("\(tokenCount) token ids, \(styleRow.count) style values")
        }
        let ids = try Self.int32(inputIds.map { Int32($0) }, shape: [1, tokenCount])
        let style = try Self.float(styleRow, shape: [1, 256])
        let timbre = try Self.float(Array(styleRow[0..<128]), shape: [1, 128])

        let prosodyResult = try predict(
            student,
            [
                "input_ids": MLFeatureValue(multiArray: ids),
                "style": MLFeatureValue(multiArray: style),
            ]
        )
        let durationHidden = try array(prosodyResult, "duration_hidden")
        let durationLogits = try array(prosodyResult, "duration_logits")
        let textFeatures = try array(prosodyResult, "text_features")
        let frames = durations(Self.values(durationLogits), tokens: tokenCount)
        let frameCount = frames.reduce(0, +)
        guard frameCount <= Self.maxFrames else {
            throw CoreMLChainError.shape("\(frameCount) frames, more than \(Self.maxFrames)")
        }

        let alignedProsody = repeatTokenRows(
            Self.values(durationHidden),
            channels: 640,
            tokens: tokenCount,
            frames: frames
        )
        let alignedText = repeatTokenColumns(
            Self.values(textFeatures),
            channels: 512,
            tokens: tokenCount,
            frames: frames
        )
        let curves = try predict(
            pitchEnergy,
            [
                "aligned": MLFeatureValue(multiArray: try Self.float(alignedProsody, shape: [1, 640, frameCount])),
                "style": MLFeatureValue(multiArray: try Self.float(Array(styleRow[128..<256]), shape: [1, 128])),
            ]
        )
        let pitch = try array(curves, "pitch")
        let energy = try array(curves, "energy")

        let source = SourceDSP.spectrum(pitch: Self.values(pitch))
        let spectrumFrames = frameCount * 120 + 1
        let generated = try predict(
            vocoder,
            [
                "text": MLFeatureValue(multiArray: try Self.float(alignedText, shape: [1, 512, frameCount])),
                "pitch": MLFeatureValue(multiArray: pitch),
                "energy": MLFeatureValue(multiArray: energy),
                "style": MLFeatureValue(multiArray: timbre),
                "spectrum": MLFeatureValue(multiArray: try Self.float(source, shape: [1, 22, spectrumFrames])),
            ]
        )
        let audio = SourceDSP.inverseSTFT(
            magnitude: Self.values(try array(generated, "magnitude")),
            phase: Self.values(try array(generated, "phase")),
            frames: spectrumFrames
        )
        guard audio.count == 600 * frameCount else {
            throw CoreMLChainError.shape("expected \(600 * frameCount) samples, got \(audio.count)")
        }
        return audio
    }

    private func durations(_ logits: [Float], tokens: Int) -> [Int] {
        (0..<tokens).map { token in
            var sum: Float = 0
            for bin in 0..<50 {
                let value = logits[token * 50 + bin]
                sum += 1 / (1 + exp(-value))
            }
            return max(1, Int(sum.rounded()))
        }
    }

    // duration_hidden is [token, channel]. Return [channel, speech frame].
    private func repeatTokenRows(
        _ input: [Float],
        channels: Int,
        tokens: Int,
        frames: [Int]
    ) -> [Float] {
        let count = frames.reduce(0, +)
        var output = [Float](repeating: 0, count: channels * count)
        var frame = 0
        for token in 0..<tokens {
            for _ in 0..<frames[token] {
                for channel in 0..<channels {
                    output[channel * count + frame] = input[token * channels + channel]
                }
                frame += 1
            }
        }
        return output
    }

    // text_features is [channel, token]. Return [channel, speech frame].
    private func repeatTokenColumns(
        _ input: [Float],
        channels: Int,
        tokens: Int,
        frames: [Int]
    ) -> [Float] {
        let count = frames.reduce(0, +)
        var output = [Float](repeating: 0, count: channels * count)
        var frame = 0
        for token in 0..<tokens {
            for _ in 0..<frames[token] {
                for channel in 0..<channels {
                    output[channel * count + frame] = input[channel * tokens + token]
                }
                frame += 1
            }
        }
        return output
    }

    // The flexible-shape packages have one default function, so no functionName.
    private static func load(_ name: String, units: MLComputeUnits = .all) throws -> MLModel {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw CoreMLChainError.missingModel(name)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    // Name the failing stage by its inputs and shapes, so a device error says which model failed.
    private func predict(_ model: MLModel, _ values: [String: MLFeatureValue]) throws
        -> MLFeatureProvider
    {
        do {
            return try model.prediction(from: MLDictionaryFeatureProvider(dictionary: values))
        } catch {
            let shapes = values.keys.sorted().map { "\($0) \(values[$0]?.multiArrayValue?.shape ?? [])" }
            let message = "prediction failed for [\(shapes.joined(separator: ", "))]: \(error)"
            throw CoreMLChainError.shape(message)
        }
    }

    private func array(_ result: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
        guard let value = result.featureValue(for: name)?.multiArrayValue else {
            throw CoreMLChainError.missingOutput(name)
        }
        return value
    }

    private static func float(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        guard values.count == shape.reduce(1, *) else {
            throw CoreMLChainError.shape("\(values.count) values do not fit \(shape)")
        }
        let array = try MLMultiArray(
            shape: shape.map(NSNumber.init),
            dataType: .float32
        )
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        values.withUnsafeBufferPointer { source in
            pointer.update(from: source.baseAddress!, count: values.count)
        }
        return array
    }

    private static func int32(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: shape.map(NSNumber.init),
            dataType: .int32
        )
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        values.withUnsafeBufferPointer { source in
            pointer.update(from: source.baseAddress!, count: values.count)
        }
        return array
    }

    // Core ML outputs may be strided. Copy in logical row-major order.
    private static func values(_ array: MLMultiArray) -> [Float] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        // The S014 student returns fp16 outputs; reading them as Float gives garbage.
        let half = array.dataType == .float16
            ? array.dataPointer.bindMemory(to: Float16.self, capacity: array.count) : nil
        var result: [Float] = []
        result.reserveCapacity(array.count)
        func visit(_ dimension: Int, _ offset: Int) {
            if dimension == shape.count {
                result.append(half.map { Float($0[offset]) } ?? pointer[offset])
                return
            }
            for index in 0..<shape[dimension] {
                visit(dimension + 1, offset + index * strides[dimension])
            }
        }
        visit(0, 0)
        return result
    }
}

enum SourceDSP {
    private static let nFFT = 20
    private static let hop = 5
    private static let harmonics = 9
    private static let upsample = 300
    private static let mergeWeight: [Float] = [
        -0.0815418661, -0.185196668, -0.182633981, -0.178372055,
        -0.0987389460, 0.0826403946, 0.0874399915, -0.390685469,
        -0.547744334,
    ]
    private static let mergeBias: Float = -0.0294502601

    static func spectrum(pitch: [Float]) -> [Float] {
        let sampleCount = pitch.count * upsample
        var random = SplitMix64(seed: 0)
        var phase = [Float](repeating: 0, count: harmonics)
        var coarse = [Float](repeating: 0, count: pitch.count * harmonics)
        for frame in 0..<pitch.count {
            for harmonic in 0..<harmonics {
                // The initial random phase in Kokoro is added before a 300x
                // downsample and does not reach the sampled centers.
                var cycles = fmodf(
                    pitch[frame] * Float(harmonic + 1) / 24_000,
                    1
                )
                if cycles < 0 { cycles += 1 }
                phase[harmonic] += cycles
                coarse[frame * harmonics + harmonic] =
                    phase[harmonic] * 2 * .pi * Float(upsample)
            }
        }

        var wave = [Float](repeating: 0, count: sampleCount)
        for sample in 0..<sampleCount {
            let coordinate = (Float(sample) + 0.5) / Float(upsample) - 0.5
            let low = max(0, min(pitch.count - 1, Int(floor(coordinate))))
            let high = max(0, min(pitch.count - 1, low + 1))
            let fraction = max(0, min(1, coordinate - Float(low)))
            let pitchFrame = min(pitch.count - 1, sample / upsample)
            let voiced = pitch[pitchFrame] > 10
            let noiseScale: Float = voiced ? 0.003 : 0.1 / 3
            var merged = mergeBias
            for harmonic in 0..<harmonics {
                let a = coarse[low * harmonics + harmonic]
                let b = coarse[high * harmonics + harmonic]
                let angle = a + (b - a) * fraction
                let sine = sin(angle) * 0.1
                let noisy = (voiced ? sine : 0) + noiseScale * random.normal()
                merged += noisy * mergeWeight[harmonic]
            }
            wave[sample] = tanh(merged)
        }
        return stft(wave)
    }

    // torch.stft: centered reflect padding, periodic Hann window, one-sided.
    private static func stft(_ wave: [Float]) -> [Float] {
        let frames = wave.count / hop + 1
        var output = [Float](repeating: 0, count: 22 * frames)
        for frame in 0..<frames {
            let start = frame * hop - nFFT / 2
            for bin in 0...nFFT / 2 {
                var real: Float = 0
                var imaginary: Float = 0
                for n in 0..<nFFT {
                    let sample = reflected(wave, start + n)
                    let window = 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(nFFT))
                    let angle = 2 * .pi * Float(bin * n) / Float(nFFT)
                    real += sample * window * cos(angle)
                    imaginary -= sample * window * sin(angle)
                }
                // A real FFT defines these two imaginary bins as positive zero.
                // Tiny sin() rounding here can otherwise flip their angle by pi.
                if bin == 0 || bin == nFFT / 2 { imaginary = 0 }
                output[bin * frames + frame] = hypot(real, imaginary)
                output[(11 + bin) * frames + frame] = atan2(imaginary, real)
            }
        }
        return output
    }

    static func inverseSTFT(
        magnitude: [Float],
        phase: [Float],
        frames: Int
    ) -> [Float] {
        let paddedCount = (frames - 1) * hop + nFFT
        var overlap = [Float](repeating: 0, count: paddedCount)
        var envelope = [Float](repeating: 0, count: paddedCount)
        for frame in 0..<frames {
            for n in 0..<nFFT {
                var value = magnitude[frame] * cos(phase[frame])
                let nyquist = 10 * frames + frame
                value += magnitude[nyquist] * cos(phase[nyquist]) * (n.isMultiple(of: 2) ? 1 : -1)
                for bin in 1..<10 {
                    let index = bin * frames + frame
                    let real = magnitude[index] * cos(phase[index])
                    let imaginary = magnitude[index] * sin(phase[index])
                    let angle = 2 * .pi * Float(bin * n) / Float(nFFT)
                    value += 2 * (real * cos(angle) - imaginary * sin(angle))
                }
                value /= Float(nFFT)
                let window = 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(nFFT))
                let index = frame * hop + n
                overlap[index] += value * window
                envelope[index] += window * window
            }
        }
        let trim = nFFT / 2
        return (trim..<(paddedCount - trim)).map { index in
            envelope[index] > 1e-11 ? overlap[index] / envelope[index] : 0
        }
    }

    private static func reflected(_ values: [Float], _ index: Int) -> Float {
        if index < 0 { return values[-index] }
        if index >= values.count { return values[2 * values.count - index - 2] }
        return values[index]
    }
}

struct SplitMix64 {
    private var state: UInt64
    private var spare: Float?

    init(seed: UInt64) { state = seed }

    mutating func uniform() -> Float {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        value ^= value >> 31
        return Float(Double(value >> 11) / Double(1 << 53))
    }

    mutating func normal() -> Float {
        if let value = spare {
            spare = nil
            return value
        }
        let u1 = max(uniform(), Float.leastNonzeroMagnitude)
        let u2 = uniform()
        let radius = sqrt(-2 * log(u1))
        let angle = 2 * Float.pi * u2
        spare = radius * sin(angle)
        return radius * cos(angle)
    }
}
