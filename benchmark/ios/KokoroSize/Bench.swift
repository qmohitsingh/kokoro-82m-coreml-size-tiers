import Foundation

// One frozen test. input_ids are speech-sound numbers, not letters.
// style is the voice row. samples is how many speaker numbers the reference has.
struct BenchChunk: Decodable, Sendable {
    let inputIds: [Int64]
    let style: [Float]
    let samples: Int

    enum CodingKeys: String, CodingKey {
        case inputIds = "input_ids"
        case style
        case samples
    }
}

struct BenchCase: Decodable, Identifiable, Sendable {
    let id: String
    let wavs: [String]
    let chunks: [BenchChunk]

    var audioSeconds: Double {
        Double(chunks.reduce(0) { $0 + $1.samples }) / 24000
    }
}

struct BenchFile: Decodable, Sendable {
    let sampleRate: Int
    let cases: [BenchCase]

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case cases
    }
}

struct StabilityReport: Encodable {
    let build: String
    let finished: Bool
    let loops: Int
    let elapsed: Double
    let engine: String
    let footprintBytes: UInt64
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
    let sampleCountMatched: Bool
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
    // Set by the XcodeGen project. S007 is the default project.yml target.
    #if S014
    static let name = "S014"
    #elseif S011
    static let name = "S011"
    #elseif S002
    static let name = "S002"
    #else
    static let name = "S007"
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

    // The graded Core ML protocol: load the fixed shapes for one case, throw
    // away 2 calls, then time 10. Model loading is outside the timed boundary.
    static func coreMLSpeed(
        _ file: BenchFile,
        tick: @escaping (String) -> Void
    ) throws -> [SpeedRow] {
        var rows: [SpeedRow] = []
        for item in file.cases {
            // One chunk's models at a time. The 30s passage is two chunks, so
            // its measured walls are added together. Loading stays outside the clock.
            var walls = [Double](repeating: 0, count: 10)
            var matched = true
            #if S014
            // The student makes its own frame count, so RTF divides by the audio it made.
            var madeSamples = [Int](repeating: 0, count: 10)
            #endif
            for (chunkIndex, chunk) in item.chunks.enumerated() {
                guard chunk.samples.isMultiple(of: 600) else { throw BenchError.badOutput }
                let loadStart = Date()
                tick("Loading Core ML models for \(item.id).")
                let chain = try CoreMLChain(
                    tokenCount: chunk.inputIds.count,
                    frameCount: chunk.samples / 600
                )
                tick(
                    String(
                        format: "%@ models loaded in %.1fs.",
                        item.id,
                        Date().timeIntervalSince(loadStart)
                    )
                )
                for warmup in 1...2 {
                    tick("\(item.id) Core ML warmup \(warmup) of 2. Not scored.")
                    _ = try autoreleasepool { try chain.synthesize(chunk) }
                }
                for run in 0..<10 {
                    let start = Date()
                    let audio = try autoreleasepool { try chain.synthesize(chunk) }
                    let partWall = Date().timeIntervalSince(start)
                    walls[run] += partWall
                    #if S014
                    madeSamples[run] += audio.count
                    #else
                    if audio.count != chunk.samples { matched = false }
                    #endif
                    tick(
                        String(
                            format: "%@ part %d run %d of 10   %.2fs",
                            item.id,
                            chunkIndex + 1,
                            run + 1,
                            partWall
                        )
                    )
                }
            }
            #if S014
            let madeSeconds = Double(madeSamples[9]) / 24000
            let rtfs = zip(walls, madeSamples).map { $0 / (Double($1) / 24000) }.sorted()
            #else
            let madeSeconds = item.audioSeconds
            let rtfs = walls.map { $0 / item.audioSeconds }.sorted()
            #endif
            let row = SpeedRow(
                build: AppBuild.name,
                id: item.id,
                audioSeconds: madeSeconds,
                medianRTF: median(rtfs),
                p90RTF: rtfs[8],
                minRTF: rtfs[0],
                maxRTF: rtfs[9],
                walls: walls,
                sampleCountMatched: matched,
                thermalState: thermal()
            )
            rows.append(row)
            try write(rows, name: "coreml-speed.json")
            tick(
                String(
                    format: "%@ Core ML done   median RTF %.3f. Saved coreml-speed.json.",
                    item.id,
                    row.medianRTF
                )
            )
        }
        return rows
    }

    // Keep making the long clip until the clock hits 60 minutes.
    // A crash or a forced close means this file never gets finished=true.
    static func stability(_ file: BenchFile, stop: StopFlag, tick: @escaping (String) -> Void) throws {
        let item = file.cases.first { $0.id == "30s" } ?? file.cases[file.cases.count - 1]
        let start = Date()
        let limit = start.addingTimeInterval(60 * 60)
        var loops = 0
        while Date() < limit {
            if stop.isStopped() { break }
            for (index, chunk) in item.chunks.enumerated() {
                if stop.isStopped() { break }
                guard chunk.samples.isMultiple(of: 600) else { throw BenchError.badOutput }
                tick("Core ML loop \(loops + 1), part \(index + 1). Loading.")
                let chain = try CoreMLChain(
                    tokenCount: chunk.inputIds.count,
                    frameCount: chunk.samples / 600
                )
                _ = try autoreleasepool { try chain.synthesize(chunk) }
            }
            if stop.isStopped() { break }
            loops += 1
            let elapsed = Date().timeIntervalSince(start)
            let note = "Core ML loop \(loops), \(Int(elapsed))s of 3600s"
            tick(note)
            try write(
                StabilityReport(
                    build: AppBuild.name,
                    finished: false,
                    loops: loops,
                    elapsed: elapsed,
                    engine: "coreml",
                    footprintBytes: MemoryFootprint.bytes()
                ),
                name: "stability.json"
            )
        }
        try write(
            StabilityReport(
                build: AppBuild.name,
                finished: !stop.isStopped() && Date() >= limit,
                loops: loops,
                elapsed: Date().timeIntervalSince(start),
                engine: "coreml",
                footprintBytes: MemoryFootprint.bytes()
            ),
            name: "stability.json"
        )
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

final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false
    func stop() { lock.lock(); stopped = true; lock.unlock() }
    func isStopped() -> Bool { lock.lock(); defer { lock.unlock() }; return stopped }
}
