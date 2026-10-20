import Foundation

// One frozen test. input_ids are speech-sound numbers, not letters, with the pad 0
// at both ends. samples is how many samples the reference has. The style row is
// the bundled af_heart row for this token count (scripts/s017_bench_inputs.py
// checked it equals the row S014's bench.json stored).
struct BenchChunk: Decodable, Sendable {
    let inputIds: [Int64]
    let samples: Int

    enum CodingKeys: String, CodingKey {
        case inputIds = "input_ids"
        case samples
    }

    func synthesize(_ chain: CoreMLChain, _ voice: VoiceTable) throws -> [Float] {
        try chain.synthesize(ids: inputIds, style: voice.style(phonemeCount: inputIds.count - 2))
    }
}

struct BenchCase: Decodable, Identifiable, Sendable {
    let id: String
    let chunks: [BenchChunk]
}

struct BenchFile: Decodable, Sendable {
    let sampleRate: Int
    let cases: [BenchCase]

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case cases
    }
}

struct SpeedRow: Encodable {
    let build: String
    let id: String
    let audioSeconds: Double
    let medianRTF: Double
    let p90RTF: Double
    let minRTF: Double
    let maxRTF: Double
    let walls: [Double]
    // Every timed run made audio, a whole number of 600-sample frames, the same length each run.
    let outputShapeValid: Bool
    let thermalState: String
}

enum BenchError: Error, CustomStringConvertible {
    case missing(String)
    case badOutput

    var description: String {
        switch self {
        case .missing(let name): return "Missing \(name) in the app bundle."
        case .badOutput: return "The model returned no audio."
        }
    }
}

enum AppBuild {
    #if S017_LISTEN
    static let name = "S017-listen"
    #else
    static let name = "S017"
    #endif
}

enum Bench {
    // Load the frozen speech-sound numbers and voice rows.
    static func load() throws -> BenchFile {
        guard let url = Bundle.main.url(forResource: "bench", withExtension: "json") else {
            throw BenchError.missing("bench.json")
        }
        return try JSONDecoder().decode(BenchFile.self, from: Data(contentsOf: url))
    }

    // The graded Core ML protocol (as S014): load the models for one chunk, throw
    // away 2 calls, then time 10. Model loading is outside the timed boundary.
    // The student makes its own frame count, so RTF divides by the audio it made.
    static func coreMLSpeed(
        _ file: BenchFile,
        tick: @escaping (String) -> Void
    ) throws -> [SpeedRow] {
        let voice = try VoiceTable()
        var rows: [SpeedRow] = []
        for item in file.cases {
            var walls = [Double](repeating: 0, count: 10)
            var madeSamples = [Int](repeating: 0, count: 10)
            var shapeValid = true
            for (chunkIndex, chunk) in item.chunks.enumerated() {
                var firstCount: Int?
                let loadStart = Date()
                tick("Loading Core ML models for \(item.id).")
                let chain = try CoreMLChain()
                tick(String(format: "%@ models loaded in %.1fs.", item.id, Date().timeIntervalSince(loadStart)))
                for warmup in 1...2 {
                    tick("\(item.id) Core ML warmup \(warmup) of 2. Not scored.")
                    _ = try autoreleasepool { try chunk.synthesize(chain, voice) }
                }
                for run in 0..<10 {
                    let start = Date()
                    let audio = try autoreleasepool { try chunk.synthesize(chain, voice) }
                    let partWall = Date().timeIntervalSince(start)
                    walls[run] += partWall
                    madeSamples[run] += audio.count
                    firstCount = firstCount ?? audio.count
                    shapeValid = shapeValid && audio.count > 0 && audio.count % 600 == 0 && audio.count == firstCount
                    tick(String(format: "%@ part %d run %d of 10   %.2fs", item.id, chunkIndex + 1, run + 1, partWall))
                }
            }
            let rtfs = zip(walls, madeSamples).map { $0 / (Double($1) / 24000) }.sorted()
            let row = SpeedRow(
                build: AppBuild.name,
                id: item.id,
                audioSeconds: Double(madeSamples[9]) / 24000,
                medianRTF: median(rtfs),
                p90RTF: rtfs[8],
                minRTF: rtfs[0],
                maxRTF: rtfs[9],
                walls: walls,
                outputShapeValid: shapeValid,
                thermalState: thermal()
            )
            rows.append(row)
            try write(rows, name: "coreml-speed.json")
            tick(String(format: "%@ Core ML done   median RTF %.3f. Saved coreml-speed.json.", item.id, row.medianRTF))
        }
        return rows
    }

    static func write(_ value: some Encodable, name: String) throws {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(name)
        let data = try JSONEncoder().encode(value)
        try data.write(to: url)
    }

    static func median(_ values: [Double]) -> Double {
        let mid = values.count / 2
        return (values[mid - 1] + values[mid]) / 2
    }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
