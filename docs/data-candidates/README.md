# Data candidates: how we picked the NASA scene

Gaussian splatting needs: (1) a camera that **moves**, not just turns; (2) a static scene;
(3) steady lighting; (4) heavy overlap and texture. Every candidate was checked against that.

| Candidate | What we checked | Verdict |
|---|---|---|
| Perseverance WATSON selfie, sol 46 | NASA raw-image API: 62 frames, camera moved ≤26 cm while turning 174° | ❌ Panorama, no parallax |
| WATSON "orbit-like" sessions (scored all 34,986 frames) | Frames showed the arm visiting several different targets | ❌ Not an orbit |
| NASA drone clips: SLS core stage at Stennis, Crew-2 Falcon 9 ([sheets](./)) | 4K, single continuous shots, partial arcs | ⚠️ Usable, judged not interesting enough |
| Astrobee `ff_nod2_bright` (ISS Node 2) | Ground-truth poses: 3 cm of movement while turning 145° | ❌ Panorama |
| Blue Origin DDL-F1 landing camera (data.nasa.gov) | 410 B&W frames at 1 Hz, narrow downward lens | ❌ Little overlap; landing obscured |
| SnowEx tundra photogrammetry (data.nasa.gov) | Catalog description | ❌ Small ground patches |
| NASA Astromaterials 3D (moon rocks) | FAQ: only meshes and CT scans are public, not raw photos | ❌ Raw photos unavailable |
| Dawn (Vesta/Ceres), MaRF, Ingenuity Navcam | Literature: lighting changes, stale data, or unverified | ❌ / not pursued |
| **Astrobee `iva_kibo_trans` (ISS Kibō)** | **11.5 m camera path, 3.8 m baseline, 1,068 frames, one session** | ✅ **Chosen** ([sheet](kibo-iva_kibo_trans.jpg)) |

Prior work: the only published Kibō splat is [arXiv 2608.21685](https://arxiv.org/html/2608.21685v1)
(five April 2021 flights, the dataset's `ff_return_journey_*` sequences; 31.2 dB PSNR). `iva_kibo_trans`
(May 2022) was not used there.
