; ModuleID = '/tmp/caxpy-99b8e5.ll'
source_filename = "/tmp/caxpy-99b8e5.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caxpy_(i64* nocapture readonly %n, i64* %ca, i64* nocapture readonly %cx, i64* nocapture readonly %incx, i64* nocapture %cy, i64* nocapture readonly %incy) local_unnamed_addr !dbg !5 {
L.entry:
  %0 = bitcast i64* %n to i32*, !dbg !13
  %cy43 = ptrtoint i64* %cy to i64
  %cy41 = bitcast i64* %cy to i8*
  %1 = load i32, i32* %0, align 4, !dbg !13, !tbaa !15
  %2 = icmp slt i32 %1, 1, !dbg !13
  br i1 %2, label %L.LB1_310, label %L.LB1_351, !dbg !13

L.LB1_351:                                        ; preds = %L.entry
  %3 = bitcast i64* %ca to i8*, !dbg !19
  %4 = tail call float (i8*, ...) bitcast (float (...)* @scabs1_ to float (i8*, ...)*)(i8* %3), !dbg !19
  %5 = fcmp oeq float %4, 0.000000e+00, !dbg !19
  br i1 %5, label %L.LB1_310, label %L.LB1_352, !dbg !19

L.LB1_352:                                        ; preds = %L.LB1_351
  %6 = bitcast i64* %incx to i32*, !dbg !20
  %7 = load i32, i32* %6, align 4, !dbg !20, !tbaa !21
  %8 = icmp eq i32 %7, 1, !dbg !20
  br i1 %8, label %L.LB1_353, label %L.LB1_317, !dbg !20

L.LB1_353:                                        ; preds = %L.LB1_352
  %9 = bitcast i64* %incy to i32*, !dbg !20
  %10 = load i32, i32* %9, align 4, !dbg !20, !tbaa !23
  %11 = icmp eq i32 %10, 1, !dbg !20
  br i1 %11, label %L.LB1_354, label %L.LB1_322thread-pre-split, !dbg !20

L.LB1_354:                                        ; preds = %L.LB1_353
  %12 = load i32, i32* %0, align 4, !dbg !25, !tbaa !15
  %13 = icmp slt i32 %12, 1, !dbg !25
  br i1 %13, label %L.LB1_310, label %L.LB1_318.preheader, !dbg !25

L.LB1_318.preheader:                              ; preds = %L.LB1_354
  %14 = bitcast i64* %ca to <{ float, float }>*
  %.elt = bitcast i64* %ca to float*
  %.unpack = load float, float* %.elt, align 1, !tbaa !26
  %.elt2 = getelementptr inbounds <{ float, float }>, <{ float, float }>* %14, i64 0, i32 1
  %.unpack3 = load float, float* %.elt2, align 1, !tbaa !26
  %15 = getelementptr i64, i64* %cx, i64 -1
  %16 = getelementptr i64, i64* %cy, i64 -1
  %17 = xor i32 %12, -1, !dbg !28
  %18 = icmp sgt i32 %17, -2, !dbg !28
  %smax = select i1 %18, i32 %17, i32 -2, !dbg !28
  %19 = add i32 %12, %smax, !dbg !28
  %20 = add i32 %19, 1, !dbg !28
  %21 = zext i32 %20 to i64, !dbg !28
  %22 = add nuw nsw i64 %21, 1, !dbg !28
  %min.iters.check = icmp ult i64 %22, 4, !dbg !28
  br i1 %min.iters.check, label %L.LB1_318.preheader61, label %vector.scevcheck, !dbg !28

vector.scevcheck:                                 ; preds = %L.LB1_318.preheader
  %23 = xor i32 %12, -1, !dbg !28
  %24 = icmp sgt i32 %23, -2, !dbg !28
  %smax40 = select i1 %24, i32 %23, i32 -2, !dbg !28
  %25 = add i32 %12, %smax40, !dbg !28
  %26 = add i32 %25, 1, !dbg !28
  %uglygep = getelementptr i8, i8* %cy41, i64 4, !dbg !28
  %uglygep42 = ptrtoint i8* %uglygep to i64
  %27 = zext i32 %26 to i64, !dbg !28
  %mul = shl nuw nsw i64 %27, 3, !dbg !28
  %28 = add i64 %mul, %uglygep42, !dbg !28
  %29 = icmp ult i64 %28, %uglygep42, !dbg !28
  %30 = zext i32 %26 to i64, !dbg !28
  %mul44 = shl nuw nsw i64 %30, 3, !dbg !28
  %31 = add i64 %mul44, %cy43, !dbg !28
  %32 = icmp ult i64 %31, %cy43, !dbg !28
  %33 = or i1 %29, %32, !dbg !28
  br i1 %33, label %L.LB1_318.preheader61, label %vector.ph, !dbg !28

vector.ph:                                        ; preds = %vector.scevcheck
  %34 = add i32 %19, 2, !dbg !28
  %35 = and i32 %34, 3, !dbg !28
  %n.mod.vf = zext i32 %35 to i64, !dbg !28
  %n.vec = sub nsw i64 %22, %n.mod.vf, !dbg !28
  %ind.end = add nsw i64 %n.vec, 1, !dbg !28
  %cast.crd = trunc i64 %n.vec to i32, !dbg !28
  %ind.end48 = sub i32 %12, %cast.crd, !dbg !28
  %broadcast.splatinsert54 = insertelement <4 x float> undef, float %.unpack, i32 0, !dbg !28
  %broadcast.splat55 = shufflevector <4 x float> %broadcast.splatinsert54, <4 x float> undef, <4 x i32> zeroinitializer, !dbg !28
  %broadcast.splatinsert56 = insertelement <4 x float> undef, float %.unpack3, i32 0, !dbg !28
  %broadcast.splat57 = shufflevector <4 x float> %broadcast.splatinsert56, <4 x float> undef, <4 x i32> zeroinitializer, !dbg !28
  br label %vector.body, !dbg !28

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %offset.idx = or i64 %index, 1
  %36 = getelementptr i64, i64* %15, i64 %offset.idx, !dbg !28
  %37 = bitcast i64* %36 to <8 x float>*, !dbg !28
  %wide.vec = load <8 x float>, <8 x float>* %37, align 1, !dbg !28, !tbaa !29
  %strided.vec = shufflevector <8 x float> %wide.vec, <8 x float> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>, !dbg !28
  %strided.vec53 = shufflevector <8 x float> %wide.vec, <8 x float> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>, !dbg !28
  %38 = fmul <4 x float> %broadcast.splat55, %strided.vec, !dbg !28
  %39 = fmul <4 x float> %broadcast.splat55, %strided.vec53, !dbg !28
  %40 = fmul <4 x float> %broadcast.splat57, %strided.vec, !dbg !28
  %41 = fmul <4 x float> %broadcast.splat57, %strided.vec53, !dbg !28
  %42 = fsub <4 x float> %38, %41, !dbg !28
  %43 = fadd <4 x float> %40, %39, !dbg !28
  %44 = getelementptr i64, i64* %16, i64 %offset.idx, !dbg !28
  %45 = bitcast i64* %44 to <{ float, float }>*, !dbg !28
  %46 = bitcast i64* %44 to <8 x float>*, !dbg !28
  %wide.vec58 = load <8 x float>, <8 x float>* %46, align 1, !dbg !28, !tbaa !31
  %strided.vec59 = shufflevector <8 x float> %wide.vec58, <8 x float> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>, !dbg !28
  %strided.vec60 = shufflevector <8 x float> %wide.vec58, <8 x float> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>, !dbg !28
  %47 = getelementptr inbounds <{ float, float }>, <{ float, float }>* %45, i64 0, i32 1, !dbg !28
  %48 = fadd <4 x float> %strided.vec59, %42, !dbg !28
  %49 = fadd <4 x float> %43, %strided.vec60, !dbg !28
  %50 = getelementptr float, float* %47, i64 -1, !dbg !28
  %51 = bitcast float* %50 to <8 x float>*, !dbg !28
  %interleaved.vec = shufflevector <4 x float> %48, <4 x float> %49, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>, !dbg !28
  store <8 x float> %interleaved.vec, <8 x float>* %51, align 4, !dbg !28, !tbaa !31
  %index.next = add i64 %index, 4
  %52 = icmp eq i64 %index.next, %n.vec
  br i1 %52, label %middle.block, label %vector.body, !llvm.loop !33

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i32 %35, 0
  br i1 %cmp.n, label %L.LB1_310, label %L.LB1_318.preheader61, !dbg !28

L.LB1_318.preheader61:                            ; preds = %middle.block, %vector.scevcheck, %L.LB1_318.preheader
  %indvars.iv.ph = phi i64 [ 1, %vector.scevcheck ], [ 1, %L.LB1_318.preheader ], [ %ind.end, %middle.block ]
  %.dY0001_320.0.ph = phi i32 [ %12, %vector.scevcheck ], [ %12, %L.LB1_318.preheader ], [ %ind.end48, %middle.block ]
  br label %L.LB1_318, !dbg !28

L.LB1_318:                                        ; preds = %L.LB1_318.preheader61, %L.LB1_318
  %indvars.iv = phi i64 [ %indvars.iv.next, %L.LB1_318 ], [ %indvars.iv.ph, %L.LB1_318.preheader61 ]
  %.dY0001_320.0 = phi i32 [ %65, %L.LB1_318 ], [ %.dY0001_320.0.ph, %L.LB1_318.preheader61 ]
  %53 = getelementptr i64, i64* %15, i64 %indvars.iv, !dbg !28
  %54 = bitcast i64* %53 to <{ float, float }>*, !dbg !28
  %.elt4 = bitcast i64* %53 to float*, !dbg !28
  %.unpack5 = load float, float* %.elt4, align 1, !dbg !28, !tbaa !29
  %.elt6 = getelementptr inbounds <{ float, float }>, <{ float, float }>* %54, i64 0, i32 1, !dbg !28
  %.unpack7 = load float, float* %.elt6, align 1, !dbg !28, !tbaa !29
  %55 = fmul float %.unpack, %.unpack5, !dbg !28
  %56 = fmul float %.unpack, %.unpack7, !dbg !28
  %57 = fmul float %.unpack3, %.unpack5, !dbg !28
  %58 = fmul float %.unpack3, %.unpack7, !dbg !28
  %59 = fsub float %55, %58, !dbg !28
  %60 = fadd float %57, %56, !dbg !28
  %61 = getelementptr i64, i64* %16, i64 %indvars.iv, !dbg !28
  %62 = bitcast i64* %61 to <{ float, float }>*, !dbg !28
  %.elt8 = bitcast i64* %61 to float*, !dbg !28
  %.unpack9 = load float, float* %.elt8, align 1, !dbg !28, !tbaa !31
  %.elt10 = getelementptr inbounds <{ float, float }>, <{ float, float }>* %62, i64 0, i32 1, !dbg !28
  %.unpack11 = load float, float* %.elt10, align 1, !dbg !28, !tbaa !31
  %63 = fadd float %.unpack9, %59, !dbg !28
  %64 = fadd float %60, %.unpack11, !dbg !28
  store float %63, float* %.elt8, align 4, !dbg !28, !tbaa !31
  store float %64, float* %.elt10, align 4, !dbg !28, !tbaa !31
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !35
  %65 = add nsw i32 %.dY0001_320.0, -1, !dbg !35
  %66 = icmp sgt i32 %.dY0001_320.0, 1, !dbg !35
  br i1 %66, label %L.LB1_318, label %L.LB1_310, !dbg !35, !llvm.loop !36

L.LB1_317:                                        ; preds = %L.LB1_352
  %67 = icmp sgt i32 %7, -1, !dbg !37
  br i1 %67, label %L.LB1_322thread-pre-split, label %L.LB1_355, !dbg !37

L.LB1_355:                                        ; preds = %L.LB1_317
  %68 = load i32, i32* %0, align 4, !dbg !37, !tbaa !15
  %69 = sub nsw i32 1, %68, !dbg !37
  %70 = mul nsw i32 %69, %7, !dbg !37
  %71 = add nsw i32 %70, 1, !dbg !37
  %phitmp = sext i32 %71 to i64
  br label %L.LB1_322

L.LB1_322thread-pre-split:                        ; preds = %L.LB1_317, %L.LB1_353
  %.pr.pr = load i32, i32* %0, align 4, !tbaa !15
  br label %L.LB1_322, !dbg !38

L.LB1_322:                                        ; preds = %L.LB1_322thread-pre-split, %L.LB1_355
  %.pr = phi i32 [ %.pr.pr, %L.LB1_322thread-pre-split ], [ %68, %L.LB1_355 ]
  %ix_307.0 = phi i64 [ 1, %L.LB1_322thread-pre-split ], [ %phitmp, %L.LB1_355 ]
  %72 = bitcast i64* %incy to i32*, !dbg !38
  %73 = load i32, i32* %72, align 4, !dbg !38, !tbaa !23
  %74 = icmp sgt i32 %73, -1, !dbg !38
  br i1 %74, label %L.LB1_323, label %L.LB1_356, !dbg !38

L.LB1_356:                                        ; preds = %L.LB1_322
  %75 = sub nsw i32 1, %.pr, !dbg !38
  %76 = mul nsw i32 %75, %73, !dbg !38
  %77 = add nsw i32 %76, 1, !dbg !38
  %phitmp37 = sext i32 %77 to i64
  br label %L.LB1_323

L.LB1_323:                                        ; preds = %L.LB1_322, %L.LB1_356
  %iy_308.0 = phi i64 [ %phitmp37, %L.LB1_356 ], [ 1, %L.LB1_322 ]
  %78 = icmp slt i32 %.pr, 1, !dbg !39
  br i1 %78, label %L.LB1_310, label %L.LB1_324.preheader, !dbg !39

L.LB1_324.preheader:                              ; preds = %L.LB1_323
  %79 = bitcast i64* %ca to <{ float, float }>*
  %.elt15 = bitcast i64* %ca to float*
  %.unpack16 = load float, float* %.elt15, align 1, !tbaa !26
  %.elt17 = getelementptr inbounds <{ float, float }>, <{ float, float }>* %79, i64 0, i32 1
  %.unpack18 = load float, float* %.elt17, align 1, !tbaa !26
  %80 = getelementptr i64, i64* %cx, i64 -1
  %81 = getelementptr i64, i64* %cy, i64 -1
  %82 = sext i32 %7 to i64, !dbg !40
  %83 = sext i32 %73 to i64, !dbg !40
  br label %L.LB1_324, !dbg !40

L.LB1_324:                                        ; preds = %L.LB1_324.preheader, %L.LB1_324
  %indvars.iv35 = phi i64 [ %iy_308.0, %L.LB1_324.preheader ], [ %indvars.iv.next36, %L.LB1_324 ]
  %indvars.iv33 = phi i64 [ %ix_307.0, %L.LB1_324.preheader ], [ %indvars.iv.next34, %L.LB1_324 ]
  %.dY0002_326.0 = phi i32 [ %.pr, %L.LB1_324.preheader ], [ %96, %L.LB1_324 ]
  %84 = getelementptr i64, i64* %80, i64 %indvars.iv33, !dbg !40
  %85 = bitcast i64* %84 to <{ float, float }>*, !dbg !40
  %.elt19 = bitcast i64* %84 to float*, !dbg !40
  %.unpack20 = load float, float* %.elt19, align 1, !dbg !40, !tbaa !29
  %.elt21 = getelementptr inbounds <{ float, float }>, <{ float, float }>* %85, i64 0, i32 1, !dbg !40
  %.unpack22 = load float, float* %.elt21, align 1, !dbg !40, !tbaa !29
  %86 = fmul float %.unpack16, %.unpack20, !dbg !40
  %87 = fmul float %.unpack16, %.unpack22, !dbg !40
  %88 = fmul float %.unpack18, %.unpack20, !dbg !40
  %89 = fmul float %.unpack18, %.unpack22, !dbg !40
  %90 = fsub float %86, %89, !dbg !40
  %91 = fadd float %88, %87, !dbg !40
  %92 = getelementptr i64, i64* %81, i64 %indvars.iv35, !dbg !40
  %93 = bitcast i64* %92 to <{ float, float }>*, !dbg !40
  %.elt23 = bitcast i64* %92 to float*, !dbg !40
  %.unpack24 = load float, float* %.elt23, align 1, !dbg !40, !tbaa !31
  %.elt25 = getelementptr inbounds <{ float, float }>, <{ float, float }>* %93, i64 0, i32 1, !dbg !40
  %.unpack26 = load float, float* %.elt25, align 1, !dbg !40, !tbaa !31
  %94 = fadd float %.unpack24, %90, !dbg !40
  %95 = fadd float %91, %.unpack26, !dbg !40
  store float %94, float* %.elt23, align 4, !dbg !40, !tbaa !31
  store float %95, float* %.elt25, align 4, !dbg !40, !tbaa !31
  %indvars.iv.next34 = add i64 %indvars.iv33, %82, !dbg !41
  %indvars.iv.next36 = add i64 %indvars.iv35, %83, !dbg !42
  %96 = add nsw i32 %.dY0002_326.0, -1, !dbg !43
  %97 = icmp sgt i32 %.dY0002_326.0, 1, !dbg !43
  br i1 %97, label %L.LB1_324, label %L.LB1_310, !dbg !43

L.LB1_310:                                        ; preds = %L.LB1_324, %L.LB1_318, %middle.block, %L.LB1_354, %L.LB1_323, %L.LB1_351, %L.entry
  ret void, !dbg !44
}

declare float @scabs1_(...) local_unnamed_addr

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "caxpy.f", directory: "/home/shoshijak/Documents/blas_ir/BLAS-3.8.0")
!4 = !{}
!5 = distinct !DISubprogram(name: "caxpy", scope: !2, file: !3, line: 89, type: !6, isLocal: false, isDefinition: true, scopeLine: 89, isOptimized: false, unit: !2, variables: !4)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !9, !10, !8, !10, !8}
!8 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIBasicType(name: "complex", size: 64, align: 32, encoding: DW_ATE_complex_float)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, align: 32, elements: !11)
!11 = !{!12}
!12 = !DISubrange(count: 0, lowerBound: 1)
!13 = !DILocation(line: 113, column: 1, scope: !14)
!14 = !DILexicalBlock(scope: !5, file: !3, line: 89, column: 1)
!15 = !{!16, !16, i64 0}
!16 = !{!"t1.2", !17, i64 0}
!17 = !{!"unlimited ptr", !18, i64 0}
!18 = !{!"Flang FAA 1"}
!19 = !DILocation(line: 114, column: 1, scope: !14)
!20 = !DILocation(line: 115, column: 1, scope: !14)
!21 = !{!22, !22, i64 0}
!22 = !{!"t1.6", !17, i64 0}
!23 = !{!24, !24, i64 0}
!24 = !{!"t1.8", !17, i64 0}
!25 = !DILocation(line: 119, column: 1, scope: !14)
!26 = !{!27, !27, i64 0}
!27 = !{!"t1.4", !17, i64 0}
!28 = !DILocation(line: 120, column: 1, scope: !14)
!29 = !{!30, !30, i64 0}
!30 = !{!"t1.c", !17, i64 0}
!31 = !{!32, !32, i64 0}
!32 = !{!"t1.f", !17, i64 0}
!33 = distinct !{!33, !34}
!34 = !{!"llvm.loop.isvectorized", i32 1}
!35 = !DILocation(line: 121, column: 1, scope: !14)
!36 = distinct !{!36, !34}
!37 = !DILocation(line: 129, column: 1, scope: !14)
!38 = !DILocation(line: 130, column: 1, scope: !14)
!39 = !DILocation(line: 131, column: 1, scope: !14)
!40 = !DILocation(line: 132, column: 1, scope: !14)
!41 = !DILocation(line: 133, column: 1, scope: !14)
!42 = !DILocation(line: 134, column: 1, scope: !14)
!43 = !DILocation(line: 135, column: 1, scope: !14)
!44 = !DILocation(line: 139, column: 1, scope: !14)
