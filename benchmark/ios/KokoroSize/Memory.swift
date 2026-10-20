import Darwin
import Foundation

// task_vm_info.phys_footprint is the app's physical memory. It does not include
// every private Neural Engine allocation.
enum MemoryFootprint {
    static func bytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<integer_t>.stride
        )
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : 0
    }
}

final class MemorySampler: @unchecked Sendable {
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "kokoro.memory")
    private var timer: DispatchSourceTimer?
    private var peak: UInt64 = 0

    func start() {
        let initial = MemoryFootprint.bytes()
        lock.lock()
        peak = initial
        timer?.cancel()
        let timer = DispatchSource.makeTimerSource(queue: queue)
        self.timer = timer
        lock.unlock()
        timer.schedule(deadline: .now(), repeating: .milliseconds(15))
        timer.setEventHandler { [weak self] in
            self?.sample()
        }
        timer.resume()
    }

    func stop() -> UInt64 {
        lock.lock()
        timer?.cancel()
        timer = nil
        lock.unlock()
        sample()
        lock.lock()
        defer { lock.unlock() }
        return peak
    }

    private func sample() {
        let value = MemoryFootprint.bytes()
        lock.lock()
        if value > peak { peak = value }
        lock.unlock()
    }
}

struct MemoryPart: Encodable {
    let id: String
    let part: Int
    let beforeLoadBytes: UInt64
    let peakLoadBytes: UInt64
    let peakSynthesisBytes: UInt64
    let afterReleaseBytes: UInt64
    let sampleCountMatched: Bool
    let thermalState: String
}

struct MemoryLoop: Encodable {
    let loop: Int
    let footprintBytes: UInt64
    let thermalState: String
}

struct MemoryReport: Encodable {
    let build: String
    let parts: [MemoryPart]
    let loops: [MemoryLoop]
}

enum MemoryDiagnostic {
    // One synthesis per chunk, then three 30-second loops. This is recorded,
    // not graded, and stays off the timed speed path.
    static func run(_ file: BenchFile, tick: (String) -> Void) throws -> MemoryReport {
        var parts: [MemoryPart] = []
        for item in file.cases {
            for (index, chunk) in item.chunks.enumerated() {
                guard chunk.samples.isMultiple(of: 600) else { throw BenchError.badOutput }
                let before = MemoryFootprint.bytes()
                tick("\(item.id) part \(index + 1): \(Self.megabytes(before)) before load.")
                let measured = try autoreleasepool { () throws -> (UInt64, UInt64, Bool) in
                    let sampler = MemorySampler()
                    sampler.start()
                    let chain = try CoreMLChain(
                        tokenCount: chunk.inputIds.count,
                        frameCount: chunk.samples / 600
                    )
                    let peakLoad = sampler.stop()
                    sampler.start()
                    let audio = try chain.synthesize(chunk)
                    let peakSynthesis = sampler.stop()
                    #if S014
                    // synthesize already checked audio.count against the student's own frames.
                    return (peakLoad, peakSynthesis, audio.count.isMultiple(of: 600))
                    #else
                    return (peakLoad, peakSynthesis, audio.count == chunk.samples)
                    #endif
                }
                let after = MemoryFootprint.bytes()
                let part = MemoryPart(
                    id: item.id,
                    part: index + 1,
                    beforeLoadBytes: before,
                    peakLoadBytes: measured.0,
                    peakSynthesisBytes: measured.1,
                    afterReleaseBytes: after,
                    sampleCountMatched: measured.2,
                    thermalState: Bench.thermal()
                )
                parts.append(part)
                tick(
                    "\(item.id) part \(index + 1): load \(Self.megabytes(measured.0)), synthesis \(Self.megabytes(measured.1)), released \(Self.megabytes(after))."
                )
            }
        }

        var loops: [MemoryLoop] = []
        if let item = file.cases.first(where: { $0.id == "30s" }) {
            for loop in 1...3 {
                for chunk in item.chunks {
                    let chain = try CoreMLChain(
                        tokenCount: chunk.inputIds.count,
                        frameCount: chunk.samples / 600
                    )
                    _ = try autoreleasepool { try chain.synthesize(chunk) }
                }
                let sample = MemoryLoop(
                    loop: loop,
                    footprintBytes: MemoryFootprint.bytes(),
                    thermalState: Bench.thermal()
                )
                loops.append(sample)
                tick("30s loop \(loop): \(Self.megabytes(sample.footprintBytes)) after release.")
            }
        }
        return MemoryReport(build: AppBuild.name, parts: parts, loops: loops)
    }

    static func megabytes(_ bytes: UInt64) -> String {
        String(format: "%.1f MB", Double(bytes) / 1_000_000)
    }
}
