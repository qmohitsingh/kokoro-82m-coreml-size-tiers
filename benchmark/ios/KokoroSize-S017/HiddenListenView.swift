#if S017_LISTEN
import AVFoundation
import SwiftUI

// Blind listening test for the S017 candidate (Listen configuration only; the
// production Release build leaves this file out). Same CoreMLChain, models,
// DSP, and synthesis settings as production. Clips live in the app's Documents,
// copied there with devicectl, never in the app bundle:
//   listen-s017/sentences.json        ids, token ids, style rows (made on the Mac)
//   listen-s017/ref/<id>.wav          original Kokoro, made on the Mac
//   listen-out-s017/s017/<id>.wav     S017 Core ML chain, made here on the phone
//   listen-out-s017/s017/frames.json  frames and checks per clip
//   listen-out-s017/make-status.json, make-error.txt
//   listen-s017.json                  one row per vote (Short hidden set)
// Four lengths set (the frozen 3s 7s 15s 30s cases; the 30s case is 30s-0 then 30s-1 as one clip):
//   listen-s017/four/ref/<case>.wav   original Kokoro, the benchmark/inputs wavs (30s joined), made on the Mac
//   listen-out-s017/four/s017/<case>.wav, frames.json   S017 chain on the phone
//   listen-s017-four.json             one row per vote, kept apart from listen-s017.json
// The screen never shows an arm name or a sentence id.

struct ListenSentence: Decodable {
    let id: String
    let inputIds: [Int64]
    let style: [Float]

    enum CodingKeys: String, CodingKey {
        case id
        case inputIds = "input_ids"
        case style
    }
}

struct ListenClipInfo: Codable {
    var frames: Int
    var samples: Int
    var peak: Double
    var rmsDb: Double
    var nonFinite: Int

    enum CodingKeys: String, CodingKey {
        case frames, samples, peak
        case rmsDb = "rms_db"
        case nonFinite = "non_finite"
    }
}

struct ListenResult: Codable {
    let listener: String
    let listenerIndex: Int
    let pairIndex: Int
    let sentenceId: String
    let arm: String
    let aIsReference: Bool
    let choice: String
    let mosA: Int
    let mosB: Int
    let timestamp: String
    // "short" or "four". Rows written before the Four lengths set existed have none.
    let set: String?
}

struct ListenPairPlan {
    let sentenceId: String
    let arm: String
    let aIsReference: Bool
    var set = "short"
}

enum ListenSet: String, CaseIterable {
    case short = "Short hidden"
    case four = "Four lengths"

    var name: String { self == .short ? "short" : "four" }
}

enum ListenFiles {
    static let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    static let dir = documents.appendingPathComponent("listen-s017")
    static let results = documents.appendingPathComponent("listen-s017.json")
    // Inputs are copied into listen-s017 by devicectl, which the app cannot write to.
    // Everything the app makes goes into listen-out-s017, a folder the app creates itself.
    static let out = documents.appendingPathComponent("listen-out-s017")
    // The Four lengths votes go to their own file, so listen-s017.json is never touched by that set.
    static let fourResults = documents.appendingPathComponent("listen-s017-four.json")

    static func results(_ set: ListenSet) -> URL { set == .short ? results : fourResults }

    static func clip(_ folder: String, _ id: String, set: String = "short") -> URL {
        let base = (folder == "s017" ? out : dir)
        return (set == "four" ? base.appendingPathComponent("four") : base)
            .appendingPathComponent(folder).appendingPathComponent("\(id).wav")
    }
}

// SplitMix64, so a listener's session is the same every time it is built.
struct ListenRNG {
    private var state: UInt64
    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }

    mutating func below(_ count: Int) -> Int { Int(next() % UInt64(count)) }

    mutating func shuffle<T>(_ items: inout [T]) {
        guard items.count > 1 else { return }
        for index in stride(from: items.count - 1, to: 0, by: -1) {
            items.swapAt(index, below(index + 1))
        }
    }
}

enum ListenPlan {
    // GOAL.md sound gate: one blind listener, the four lengths plus a few unseen sentences.
    static let pairsPerListener = 8
    // The S014 sessions heard entries 0-31 (int4 pilot 0-23, mixed8 24-31); start after them.
    static let firstEntry = 32
    // S017 listener 0 heard entries 32-39. Later listeners get short clips only, from entry 40 on.
    static let shortFirstEntry = 40
    static let shortSeconds = 2.0...4.0

    // One fixed shuffled order of all sentences (the file is grouped by category).
    // Listener 0 took the 8 entries starting at 32. Listener k >= 1 takes the k-th group of 8 entries, from entry 40
    // on, whose reference and S017 clips both last 2-4 s (picked by clip length only, never by text). Every pair is
    // original vs S017. The reference sits on side A in exactly half the pairs, in an order seeded by k.
    static func session(listenerIndex: Int, sentenceIds: [String],
                        seconds: (String) -> (ref: Double, s017: Double)?) -> [ListenPairPlan] {
        var order = sentenceIds
        var sentenceRNG = ListenRNG(seed: 0x5014)
        sentenceRNG.shuffle(&order)
        let picked: [String]
        if listenerIndex == 0 {
            let start = firstEntry % order.count
            picked = (0..<pairsPerListener).map { order[(start + $0) % order.count] }
        } else {
            let short = order.dropFirst(shortFirstEntry).filter { id in
                guard let s = seconds(id) else { return false }
                return shortSeconds.contains(s.ref) && shortSeconds.contains(s.s017)
            }
            let start = (listenerIndex - 1) * pairsPerListener
            guard start + pairsPerListener <= short.count else { return [] }
            picked = Array(short[start..<(start + pairsPerListener)])
        }
        var sides = Array(repeating: true, count: pairsPerListener / 2)
            + Array(repeating: false, count: pairsPerListener / 2)
        var rng = ListenRNG(seed: 0xA5_0000 + UInt64(listenerIndex))
        rng.shuffle(&sides)
        return zip(picked, sides).map { id, aIsReference in
            ListenPairPlan(sentenceId: id, arm: "s017", aIsReference: aIsReference)
        }
    }

    // The four frozen lengths in length order, original vs S017, the reference on side A in exactly
    // two of the four pairs, in an order seeded by the listener index.
    static let fourCases = ["3s", "7s", "15s", "30s"]

    static func fourLengths(listenerIndex: Int) -> [ListenPairPlan] {
        var sides = [true, true, false, false]
        var rng = ListenRNG(seed: 0xF0_0000 + UInt64(listenerIndex))
        rng.shuffle(&sides)
        return zip(fourCases, sides).map { id, aIsReference in
            ListenPairPlan(sentenceId: id, arm: "s017", aIsReference: aIsReference, set: "four")
        }
    }
}

// Makes the S017 arm with the same CoreMLChain the speed test uses.
enum HiddenListenMaker {
    static func run(tick: @escaping (String) -> Void) throws {
        let sentences = try JSONDecoder().decode(
            [ListenSentence].self,
            from: Data(contentsOf: ListenFiles.dir.appendingPathComponent("sentences.json"))
        )
        let out = ListenFiles.out.appendingPathComponent("s017")
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let infoURL = out.appendingPathComponent("frames.json")
        var infos = (try? JSONDecoder().decode(
            [String: ListenClipInfo].self, from: Data(contentsOf: infoURL)
        )) ?? [:]
        var chain: CoreMLChain?
        for (index, sentence) in sentences.enumerated() {
            let wav = ListenFiles.clip("s017", sentence.id)
            if FileManager.default.fileExists(atPath: wav.path), infos[sentence.id] != nil {
                tick("\(index + 1)/\(sentences.count)")
                continue
            }
            if chain == nil {
                tick("Loading Core ML models... \(index)/\(sentences.count)")
                chain = try CoreMLChain()
            }
            let current = chain!
            let audio = try autoreleasepool { try current.synthesize(ids: sentence.inputIds, style: sentence.style) }
            var peak: Double = 0
            var squares: Double = 0
            var bad = 0
            var clean = audio
            for position in clean.indices {
                if !clean[position].isFinite {
                    bad += 1
                    clean[position] = 0
                }
                peak = max(peak, Double(abs(clean[position])))
                squares += Double(clean[position]) * Double(clean[position])
            }
            let rms = (squares / Double(max(1, clean.count))).squareRoot()
            try wavData(clean).write(to: wav, options: .atomic)
            infos[sentence.id] = ListenClipInfo(
                frames: audio.count / 600,
                samples: audio.count,
                peak: peak,
                rmsDb: 20 * log10(max(rms, 1e-9)),
                nonFinite: bad
            )
            try JSONEncoder().encode(infos).write(to: infoURL, options: .atomic)
            tick("\(index + 1)/\(sentences.count)")
        }
        try runFour(chain: &chain, tick: tick)
    }

    // Four lengths set: the frozen 3s, 7s, 15s and 30s cases, from the same bundled bench.json and
    // af_heart rows as the speed test. The 30s case is its two chunks joined in bench order.
    static func runFour(chain: inout CoreMLChain?, tick: @escaping (String) -> Void) throws {
        let out = ListenFiles.out.appendingPathComponent("four/s017")
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let infoURL = out.appendingPathComponent("frames.json")
        var infos = (try? JSONDecoder().decode(
            [String: ListenClipInfo].self, from: Data(contentsOf: infoURL)
        )) ?? [:]
        let voice = try VoiceTable()
        let cases = try Bench.load().cases
        for (index, item) in cases.enumerated() {
            let wav = ListenFiles.clip("s017", item.id, set: "four")
            if FileManager.default.fileExists(atPath: wav.path), infos[item.id] != nil {
                tick("four \(index + 1)/\(cases.count)")
                continue
            }
            if chain == nil {
                tick("Loading Core ML models... four")
                chain = try CoreMLChain()
            }
            let current = chain!
            var audio: [Float] = []
            for chunk in item.chunks {
                audio += try autoreleasepool { try chunk.synthesize(current, voice) }
            }
            var peak: Double = 0
            var squares: Double = 0
            var bad = 0
            var clean = audio
            for position in clean.indices {
                if !clean[position].isFinite {
                    bad += 1
                    clean[position] = 0
                }
                peak = max(peak, Double(abs(clean[position])))
                squares += Double(clean[position]) * Double(clean[position])
            }
            let rms = (squares / Double(max(1, clean.count))).squareRoot()
            try wavData(clean).write(to: wav, options: .atomic)
            infos[item.id] = ListenClipInfo(
                frames: audio.count / 600,
                samples: audio.count,
                peak: peak,
                rmsDb: 20 * log10(max(rms, 1e-9)),
                nonFinite: bad
            )
            try JSONEncoder().encode(infos).write(to: infoURL, options: .atomic)
            tick("four \(index + 1)/\(cases.count)")
        }
    }

    // Documents/listen-out-s017/make-status.json says how far it got, and why it stopped.
    static func start(tick: @escaping (String) -> Void, finish: @escaping (String) -> Void) {
        Task.detached {
            try? FileManager.default.createDirectory(at: ListenFiles.out, withIntermediateDirectories: true)
            let statusURL = ListenFiles.out.appendingPathComponent("make-status.json")
            func note(_ text: String, done: Bool, error: Bool) {
                let body: [String: Any] = ["text": text, "done": done, "error": error]
                if let data = try? JSONSerialization.data(withJSONObject: body) {
                    try? data.write(to: statusURL, options: .atomic)
                }
            }
            do {
                try run { text in
                    note(text, done: false, error: false)
                    tick(text)
                }
                note("done", done: true, error: false)
                finish("S017 clips ready.")
            } catch {
                let text = String(describing: error)
                note(text, done: true, error: true)
                try? text.write(
                    to: ListenFiles.out.appendingPathComponent("make-error.txt"),
                    atomically: true, encoding: .utf8
                )
                finish("Making clips failed: \(text)")
            }
        }
    }

    // 24 kHz, 16-bit, mono, no loudness change.
    static func wavData(_ samples: [Float]) -> Data {
        var data = Data()
        func u32(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ value: UInt16) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        let bytes = UInt32(samples.count * 2)
        data.append("RIFF".data(using: .ascii)!)
        u32(36 + bytes)
        data.append("WAVEfmt ".data(using: .ascii)!)
        u32(16); u16(1); u16(1); u32(24000); u32(48000); u16(2); u16(16)
        data.append("data".data(using: .ascii)!)
        u32(bytes)
        for sample in samples {
            let scaled = (max(-1, min(1, sample)) * 32767).rounded()
            u16(UInt16(bitPattern: Int16(scaled)))
        }
        return data
    }
}

@MainActor
final class HiddenListenModel: ObservableObject {
    @Published var makeStatus = "Not started."
    @Published var making = false
    @Published var listener = ""
    @Published var selectedSet = ListenSet.short
    @Published var message = ""
    @Published var session: [ListenPairPlan] = []
    @Published var pairIndex = 0
    @Published var finished = false
    @Published var choice: String?
    @Published var mosA: Int?
    @Published var mosB: Int?
    @Published var heardA = false
    @Published var heardB = false

    private var listenerName = ""
    private var listenerIndex = 0
    private var results: [ListenResult] = []
    private var resultsURL = ListenFiles.results
    private var setName = "short"
    private var playToken: UUID?
    private let player = ListenPlayer()

    var canVote: Bool { heardA && heardB }
    var canNext: Bool { canVote && choice != nil && mosA != nil && mosB != nil }
    var started: Bool { !session.isEmpty }

    func makeClips() {
        making = true
        UIApplication.shared.isIdleTimerDisabled = true
        makeStatus = "Starting..."
        HiddenListenMaker.start(
            tick: { text in Task { @MainActor in self.makeStatus = text } },
            finish: { text in
                Task { @MainActor in
                    self.making = false
                    self.makeStatus = text
                    UIApplication.shared.isIdleTimerDisabled = false
                }
            }
        )
    }

    private func loadResults(_ url: URL) -> [ListenResult]? {
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        return try? JSONDecoder().decode([ListenResult].self, from: Data(contentsOf: url))
    }

    func startSession() {
        let name = listener.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { message = "Type a listener name first."; return }
        let url = ListenFiles.results(selectedSet)
        guard let existing = loadResults(url) else {
            message = "\(url.lastPathComponent) cannot be read. Not starting, so no votes are overwritten."
            return
        }
        guard let data = try? Data(contentsOf: ListenFiles.dir.appendingPathComponent("sentences.json")),
              let sentences = try? JSONDecoder().decode([ListenSentence].self, from: data)
        else { message = "listen-s017/sentences.json is missing."; return }

        let mine = existing.filter { $0.listener.lowercased() == name.lowercased() }
        let names = Set(existing.map { $0.listener.lowercased() })
        let index = mine.first?.listenerIndex ?? names.count
        let plan = selectedSet == .four
            ? ListenPlan.fourLengths(listenerIndex: index)
            : ListenPlan.session(listenerIndex: index, sentenceIds: sentences.map(\.id)) { id in
                guard let ref = Self.seconds(ListenFiles.clip("ref", id)),
                      let s017 = Self.seconds(ListenFiles.clip("s017", id)) else { return nil }
                return (ref, s017)
            }
        guard !plan.isEmpty else { message = "Not enough unheard 2-4 s sentences left."; return }

        let missing = plan.filter { pair in
            !FileManager.default.fileExists(atPath: ListenFiles.clip("ref", pair.sentenceId, set: pair.set).path)
                || !FileManager.default.fileExists(
                    atPath: ListenFiles.clip(pair.arm, pair.sentenceId, set: pair.set).path)
        }.count
        guard missing == 0 else {
            message = "\(missing) clips are missing. Copy the Mac clips and tap Make S017 clips first."
            return
        }
        let done = Set(mine.map(\.pairIndex))
        guard let next = plan.indices.first(where: { !done.contains($0) }) else {
            message = "This listener has finished. Use a new name."
            return
        }
        results = existing
        resultsURL = url
        setName = selectedSet.name
        listenerName = mine.first?.listener ?? name
        listenerIndex = index
        session = plan
        pairIndex = next
        finished = false
        choice = nil; mosA = nil; mosB = nil
        heardA = false; heardB = false
        message = ""
    }

    private static func seconds(_ url: URL) -> Double? {
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        return Double(file.length) / file.fileFormat.sampleRate
    }

    private func samples(side: String) -> [Float] {
        let pair = session[pairIndex]
        let wantReference = (side == "A") == pair.aIsReference
        let url = ListenFiles.clip(wantReference ? "ref" : pair.arm, pair.sentenceId, set: pair.set)
        guard let file = try? AVAudioFile(forReading: url),
              let buffer = AVAudioPCMBuffer(
                  pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
              (try? file.read(into: buffer)) != nil,
              let channel = buffer.floatChannelData?[0]
        else { return [] }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }

    func play(_ side: String) {
        guard started, !finished else { return }
        let audio = samples(side: side)
        guard !audio.isEmpty else { message = "Could not read that clip."; return }
        // A clip counts as heard only when it plays to the end. Stopping it early (by playing the
        // other side) also fires the handler, so check the token and the elapsed time.
        let token = UUID()
        playToken = token
        let start = Date()
        let length = Double(audio.count) / 24000
        player.play(samples: audio) { [weak self] in
            Task { @MainActor in
                guard let self, self.playToken == token,
                      Date().timeIntervalSince(start) >= length - 0.1 else { return }
                if side == "A" { self.heardA = true } else { self.heardB = true }
            }
        }
    }

    func next() {
        guard canNext, let choice, let mosA, let mosB else { return }
        let pair = session[pairIndex]
        results.append(ListenResult(
            listener: listenerName,
            listenerIndex: listenerIndex,
            pairIndex: pairIndex,
            sentenceId: pair.sentenceId,
            arm: pair.arm,
            aIsReference: pair.aIsReference,
            choice: choice,
            mosA: mosA,
            mosB: mosB,
            timestamp: ISO8601DateFormatter().string(from: Date()),
            set: setName
        ))
        // Saved before moving on, so a killed app loses nothing already submitted.
        do {
            try JSONEncoder().encode(results).write(to: resultsURL, options: .atomic)
            message = ""
        } catch {
            message = "Save failed: \(error)"
            results.removeLast()
            return
        }
        self.choice = nil; self.mosA = nil; self.mosB = nil
        heardA = false; heardB = false; playToken = nil
        if pairIndex + 1 >= session.count {
            finished = true
        } else {
            pairIndex += 1
        }
    }

    func reset() {
        session = []
        finished = false
        listener = ""
        message = ""
    }
}

struct HiddenListenView: View {
    @StateObject private var model = HiddenListenModel()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Button("Make S017 clips") { model.makeClips() }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.making)
                Text(model.makeStatus).font(.footnote).foregroundStyle(.secondary)
                Divider()
                Picker("Set", selection: $model.selectedSet) {
                    ForEach(ListenSet.allCases, id: \.self) { Text($0.rawValue) }
                }
                .pickerStyle(.segmented)
                .disabled(model.started)
                TextField("Listener name", text: $model.listener)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .disabled(model.started)
                if !model.started {
                    Button("Start session") { model.startSession() }.buttonStyle(.bordered)
                } else if model.finished {
                    Text("All \(model.session.count) pairs done. Thank you.").font(.headline)
                    Button("New listener") { model.reset() }.buttonStyle(.bordered)
                } else {
                    pairBody
                }
                if !model.message.isEmpty {
                    Text(model.message).font(.footnote).foregroundStyle(.red)
                }
            }
            .padding()
        }
        .navigationTitle("Hidden listening")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var pairBody: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Sentence \(model.pairIndex + 1)/\(model.session.count)"
                 + (model.canVote ? "" : ", play A and B to the end to vote")).font(.headline)
            HStack {
                wide("Play A", selected: model.heardA) { model.play("A") }
                wide("Play B", selected: model.heardB) { model.play("B") }
            }
            Group {
                HStack {
                    wide("Prefer A", selected: model.choice == "A") { model.choice = "A" }
                    wide("Prefer B", selected: model.choice == "B") { model.choice = "B" }
                    wide("Tie", selected: model.choice == "tie") { model.choice = "tie" }
                }
                Text("A quality, 1 bad to 5 excellent").font(.footnote)
                HStack {
                    ForEach(1...5, id: \.self) { score in
                        wide("\(score)", selected: model.mosA == score) { model.mosA = score }
                    }
                }
                Text("B quality, 1 bad to 5 excellent").font(.footnote)
                HStack {
                    ForEach(1...5, id: \.self) { score in
                        wide("\(score)", selected: model.mosB == score) { model.mosB = score }
                    }
                }
            }
            .disabled(!model.canVote)
            Button("Next") { model.next() }
                .buttonStyle(.borderedProminent)
                .frame(maxWidth: .infinity)
                .disabled(!model.canNext)
        }
    }

    @ViewBuilder
    private func wide(_ title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        if selected {
            Button(action: action) { Text(title).frame(maxWidth: .infinity) }
                .buttonStyle(.borderedProminent)
        } else {
            Button(action: action) { Text(title).frame(maxWidth: .infinity) }
                .buttonStyle(.bordered)
        }
    }
}

// AudioPlayer with a played-to-end callback, kept here so production code is unchanged.
final class ListenPlayer {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!

    init() {
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        try? AVAudioSession.sharedInstance().setCategory(.playback)
        try? AVAudioSession.sharedInstance().setActive(true)
        try? engine.start()
    }

    func play(samples: [Float], done: @escaping @Sendable () -> Void) {
        guard !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
        else { return }
        buffer.frameLength = buffer.frameCapacity
        samples.withUnsafeBufferPointer { src in
            buffer.floatChannelData?[0].update(from: src.baseAddress!, count: samples.count)
        }
        node.stop()
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in done() }
        node.play()
    }
}
#endif
