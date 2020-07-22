
<root>
8; undefined functionB 

embedding_key

	undefined
:; undefined function B 

embedding_key

	undefined
:; undefined function B 

embedding_key

	undefined
7alloca B+
)
	full_text

%1 = alloca i32, align 4
@alloca B4
2
	full_text%
#
!%2 = alloca [300 x i32], align 16
;bitcast B.
,
	full_text

%3 = bitcast i32* %1 to i8*
&i32* B

	full_text
	
i32* %1
Zcall BP
N
	full_textA
?
=call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %3) #2
$i8* B

	full_text


i8* %3
Cbitcast B6
4
	full_text'
%
#%4 = bitcast [300 x i32]* %2 to i8*
6[300 x i32]* B"
 
	full_text

[300 x i32]* %2
]call BS
Q
	full_textD
B
@call void @llvm.lifetime.start.p0i8(i64 1200, i8* nonnull %4) #2
$i8* B

	full_text


i8* %4
ïcall Bä
á
	full_textz
x
v%5 = call i32 (i8*, ...) @scanf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i32* nonnull %1)
&i32* B

	full_text
	
i32* %1
Fload B<
:
	full_text-
+
)%6 = load i32, i32* %1, align 4, !tbaa !2
&i32* B

	full_text
	
i32* %1
4icmp B*
(
	full_text

%7 = icmp sgt i32 %6, 0
$i32 B

	full_text


i32 %6
8br B0
.
	full_text!

br i1 %7, label %8, label %85
"i1 B

	full_text	

i1 %7
(br 8B

	full_text

br label %9
Bphi 8B7
5
	full_text(
&
$%10 = phi i64 [ %13, %9 ], [ 0, %8 ]
'i64 8B

	full_text
	
i64 %13
qgetelementptr 8B\
Z
	full_textM
K
I%11 = getelementptr inbounds [300 x i32], [300 x i32]* %2, i64 0, i64 %10
8[300 x i32]* 8B"
 
	full_text

[300 x i32]* %2
'i64 8B

	full_text
	
i64 %10
ôcall 8Bå
â
	full_text|
z
x%12 = call i32 (i8*, ...) @scanf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i32* nonnull %11)
)i32* 8B

	full_text


i32* %11
:add 8B/
-
	full_text 

%13 = add nuw nsw i64 %10, 1
'i64 8B

	full_text
	
i64 %10
Iload 8B=
;
	full_text.
,
*%14 = load i32, i32* %1, align 4, !tbaa !2
(i32* 8B

	full_text
	
i32* %1
8sext 8B,
*
	full_text

%15 = sext i32 %14 to i64
'i32 8B

	full_text
	
i32 %14
:icmp 8B.
,
	full_text

%16 = icmp slt i64 %13, %15
'i64 8B

	full_text
	
i64 %13
'i64 8B

	full_text
	
i64 %15
;br 8B1
/
	full_text"
 
br i1 %16, label %9, label %17
%i1 8B

	full_text


i1 %16
8icmp 8B,
*
	full_text

%18 = icmp sgt i32 %14, 1
'i32 8B

	full_text
	
i32 %14
<br 8B2
0
	full_text#
!
br i1 %18, label %19, label %64
%i1 8B

	full_text


i1 %18
7add 8B,
*
	full_text

%20 = add nsw i32 %14, -1
'i32 8B

	full_text
	
i32 %14
8sext 8B,
*
	full_text

%21 = sext i32 %14 to i64
'i32 8B

	full_text
	
i32 %14
8zext 8B,
*
	full_text

%22 = zext i32 %14 to i64
'i32 8B

	full_text
	
i32 %14
8zext 8B,
*
	full_text

%23 = zext i32 %20 to i64
'i32 8B

	full_text
	
i32 %20
:add 8B/
-
	full_text 

%24 = add nuw nsw i64 %22, 1
'i64 8B

	full_text
	
i64 %22
7add 8B,
*
	full_text

%25 = add nsw i64 %22, -2
'i64 8B

	full_text
	
i64 %22
)br 8B

	full_text

br label %26
Dphi 8B9
7
	full_text*
(
&%27 = phi i64 [ 0, %19 ], [ %29, %61 ]
'i64 8B

	full_text
	
i64 %29
Dphi 8B9
7
	full_text*
(
&%28 = phi i64 [ 1, %19 ], [ %62, %61 ]
'i64 8B

	full_text
	
i64 %62
:add 8B/
-
	full_text 

%29 = add nuw nsw i64 %27, 1
'i64 8B

	full_text
	
i64 %27
:icmp 8B.
,
	full_text

%30 = icmp slt i64 %29, %21
'i64 8B

	full_text
	
i64 %29
'i64 8B

	full_text
	
i64 %21
<br 8B2
0
	full_text#
!
br i1 %30, label %31, label %61
%i1 8B

	full_text


i1 %30
8sub 8B-
+
	full_text

%32 = sub nsw i64 %24, %27
'i64 8B

	full_text
	
i64 %24
'i64 8B

	full_text
	
i64 %27
qgetelementptr 8B\
Z
	full_textM
K
I%33 = getelementptr inbounds [300 x i32], [300 x i32]* %2, i64 0, i64 %27
8[300 x i32]* 8B"
 
	full_text

[300 x i32]* %2
'i64 8B

	full_text
	
i64 %27
2and 8B'
%
	full_text

%34 = and i64 %32, 1
'i64 8B

	full_text
	
i64 %32
7icmp 8B+
)
	full_text

%35 = icmp eq i64 %34, 0
'i64 8B

	full_text
	
i64 %34
<br 8B2
0
	full_text#
!
br i1 %35, label %44, label %36
%i1 8B

	full_text


i1 %35
Jload 8B>
<
	full_text/
-
+%37 = load i32, i32* %33, align 4, !tbaa !2
)i32* 8B

	full_text


i32* %33
qgetelementptr 8B\
Z
	full_textM
K
I%38 = getelementptr inbounds [300 x i32], [300 x i32]* %2, i64 0, i64 %28
8[300 x i32]* 8B"
 
	full_text

[300 x i32]* %2
'i64 8B

	full_text
	
i64 %28
Jload 8B>
<
	full_text/
-
+%39 = load i32, i32* %38, align 4, !tbaa !2
)i32* 8B

	full_text


i32* %38
9icmp 8B-
+
	full_text

%40 = icmp eq i32 %37, %39
'i32 8B

	full_text
	
i32 %37
'i32 8B

	full_text
	
i32 %39
<br 8B2
0
	full_text#
!
br i1 %40, label %41, label %42
%i1 8B

	full_text


i1 %40
Hstore 8B;
9
	full_text,
*
(store i32 0, i32* %38, align 4, !tbaa !2
)i32* 8B

	full_text


i32* %38
)br 8B

	full_text

br label %42
:add 8	B/
-
	full_text 

%43 = add nuw nsw i64 %28, 1
'i64 8	B

	full_text
	
i64 %28
)br 8	B

	full_text

br label %44
Fphi 8
B;
9
	full_text,
*
(%45 = phi i64 [ %43, %42 ], [ %28, %31 ]
'i64 8
B

	full_text
	
i64 %43
'i64 8
B

	full_text
	
i64 %28
9icmp 8
B-
+
	full_text

%46 = icmp eq i64 %25, %27
'i64 8
B

	full_text
	
i64 %25
'i64 8
B

	full_text
	
i64 %27
<br 8
B2
0
	full_text#
!
br i1 %46, label %61, label %47
%i1 8
B

	full_text


i1 %46
)br 8B

	full_text

br label %48
Fphi 8B;
9
	full_text,
*
(%49 = phi i64 [ %45, %47 ], [ %88, %87 ]
'i64 8B

	full_text
	
i64 %45
'i64 8B

	full_text
	
i64 %88
Jload 8B>
<
	full_text/
-
+%50 = load i32, i32* %33, align 4, !tbaa !2
)i32* 8B

	full_text


i32* %33
qgetelementptr 8B\
Z
	full_textM
K
I%51 = getelementptr inbounds [300 x i32], [300 x i32]* %2, i64 0, i64 %49
8[300 x i32]* 8B"
 
	full_text

[300 x i32]* %2
'i64 8B

	full_text
	
i64 %49
Jload 8B>
<
	full_text/
-
+%52 = load i32, i32* %51, align 4, !tbaa !2
)i32* 8B

	full_text


i32* %51
9icmp 8B-
+
	full_text

%53 = icmp eq i32 %50, %52
'i32 8B

	full_text
	
i32 %50
'i32 8B

	full_text
	
i32 %52
<br 8B2
0
	full_text#
!
br i1 %53, label %54, label %55
%i1 8B

	full_text


i1 %53
Hstore 8B;
9
	full_text,
*
(store i32 0, i32* %51, align 4, !tbaa !2
)i32* 8B

	full_text


i32* %51
)br 8B

	full_text

br label %55
:add 8B/
-
	full_text 

%56 = add nuw nsw i64 %49, 1
'i64 8B

	full_text
	
i64 %49
Jload 8B>
<
	full_text/
-
+%57 = load i32, i32* %33, align 4, !tbaa !2
)i32* 8B

	full_text


i32* %33
qgetelementptr 8B\
Z
	full_textM
K
I%58 = getelementptr inbounds [300 x i32], [300 x i32]* %2, i64 0, i64 %56
8[300 x i32]* 8B"
 
	full_text

[300 x i32]* %2
'i64 8B

	full_text
	
i64 %56
Jload 8B>
<
	full_text/
-
+%59 = load i32, i32* %58, align 4, !tbaa !2
)i32* 8B

	full_text


i32* %58
9icmp 8B-
+
	full_text

%60 = icmp eq i32 %57, %59
'i32 8B

	full_text
	
i32 %57
'i32 8B

	full_text
	
i32 %59
<br 8B2
0
	full_text#
!
br i1 %60, label %86, label %87
%i1 8B

	full_text


i1 %60
:add 8B/
-
	full_text 

%62 = add nuw nsw i64 %28, 1
'i64 8B

	full_text
	
i64 %28
9icmp 8B-
+
	full_text

%63 = icmp eq i64 %29, %23
'i64 8B

	full_text
	
i64 %29
'i64 8B

	full_text
	
i64 %23
<br 8B2
0
	full_text#
!
br i1 %63, label %64, label %26
%i1 8B

	full_text


i1 %63
8icmp 8B,
*
	full_text

%65 = icmp sgt i32 %14, 0
'i32 8B

	full_text
	
i32 %14
<br 8B2
0
	full_text#
!
br i1 %65, label %66, label %85
%i1 8B

	full_text


i1 %65
)br 8B

	full_text

br label %67
Dphi 8B9
7
	full_text*
(
&%68 = phi i64 [ %81, %79 ], [ 0, %66 ]
'i64 8B

	full_text
	
i64 %81
Ephi 8B:
8
	full_text+
)
'%69 = phi i32 [ %80, %79 ], [ -1, %66 ]
'i32 8B

	full_text
	
i32 %80
8icmp 8B,
*
	full_text

%70 = icmp eq i32 %69, -1
'i32 8B

	full_text
	
i32 %69
qgetelementptr 8B\
Z
	full_textM
K
I%71 = getelementptr inbounds [300 x i32], [300 x i32]* %2, i64 0, i64 %68
8[300 x i32]* 8B"
 
	full_text

[300 x i32]* %2
'i64 8B

	full_text
	
i64 %68
Jload 8B>
<
	full_text/
-
+%72 = load i32, i32* %71, align 4, !tbaa !2
)i32* 8B

	full_text


i32* %71
<br 8B2
0
	full_text#
!
br i1 %70, label %73, label %75
%i1 8B

	full_text


i1 %70
ëcall 8BÑ
Å
	full_textt
r
p%74 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i32 %72)
'i32 8B

	full_text
	
i32 %72
)br 8B

	full_text

br label %79
7icmp 8B+
)
	full_text

%76 = icmp eq i32 %72, 0
'i32 8B

	full_text
	
i32 %72
<br 8B2
0
	full_text#
!
br i1 %76, label %79, label %77
%i1 8B

	full_text


i1 %76
ìcall 8BÜ
É
	full_textv
t
r%78 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0), i32 %72)
'i32 8B

	full_text
	
i32 %72
)br 8B

	full_text

br label %79
Rphi 8BG
E
	full_text8
6
4%80 = phi i32 [ 0, %73 ], [ %69, %77 ], [ %69, %75 ]
'i32 8B

	full_text
	
i32 %69
'i32 8B

	full_text
	
i32 %69
:add 8B/
-
	full_text 

%81 = add nuw nsw i64 %68, 1
'i64 8B

	full_text
	
i64 %68
Iload 8B=
;
	full_text.
,
*%82 = load i32, i32* %1, align 4, !tbaa !2
(i32* 8B

	full_text
	
i32* %1
8sext 8B,
*
	full_text

%83 = sext i32 %82 to i64
'i32 8B

	full_text
	
i32 %82
:icmp 8B.
,
	full_text

%84 = icmp slt i64 %81, %83
'i64 8B

	full_text
	
i64 %81
'i64 8B

	full_text
	
i64 %83
<br 8B2
0
	full_text#
!
br i1 %84, label %67, label %85
%i1 8B

	full_text


i1 %84
]call 8BQ
O
	full_textB
@
>call void @llvm.lifetime.end.p0i8(i64 1200, i8* nonnull %4) #2
&i8* 8B

	full_text


i8* %4
Zcall 8BN
L
	full_text?
=
;call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %3) #2
&i8* 8B

	full_text


i8* %3
'ret 8B

	full_text

	ret i32 0
Hstore 8B;
9
	full_text,
*
(store i32 0, i32* %58, align 4, !tbaa !2
)i32* 8B

	full_text


i32* %58
)br 8B

	full_text

br label %87
6add 8B+
)
	full_text

%88 = add nsw i64 %49, 2
'i64 8B

	full_text
	
i64 %49
9icmp 8B-
+
	full_text

%89 = icmp eq i64 %88, %22
'i64 8B

	full_text
	
i64 %88
'i64 8B

	full_text
	
i64 %22
<br 8B2
0
	full_text#
!
br i1 %89, label %61, label %48
%i1 8B

	full_text


i1 %89
:; undefined function B 

embedding_key

	undefined
:; undefined function B 

embedding_key

	undefined
:; undefined function B 

embedding_key

	undefined
:; undefined function B 

embedding_key

	undefined
}call 8Bq
o
	full_textb
`
^tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
øcall 8B≤
Ø
	full_text°
û
õ%1 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #2
&ret 8B

	full_text


ret void
di8*8BY
W
	full_textJ
H
Fi8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0)
Gi8*8B<
:
	full_text-
+
)@__dso_handle = external hidden global i8
$i328B

	full_text


i32 -1
bi8*8BW
U
	full_textH
F
Di8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0)
#i328B

	full_text	

i32 0
&i648B

	full_text


i64 1200
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 1
âvoid (i8*)*8Bv
t
	full_textg
e
cvoid (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*)
$i648B

	full_text


i64 -2
#i648B

	full_text	

i64 4
ystruct*8Bj
h
	full_text[
Y
W@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 2
ñi8*8Bä
á
	full_textz
x
vi8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0)    	  
 

                    !    "# "" $% $& $$ '( '* )) +, +. -- /0 // 12 11 34 33 56 55 78 77 9; :: <= << >? >> @A @B @@ CD CF EG EE HI HJ HH KL KK MN MM OP OR QQ ST SU SS VW VV XY XZ XX [\ [^ ]] _a `` bd ce cc fg fh ff ij im ln ll op oo qr qs qq tu tt vw vx vv yz y| {{ } ~~ ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ áà á
â áá äã äç åå éè é
ê éé ëí ëî ìì ïñ ïô òò öõ öö úù úú ûü û
† ûû °¢ °° £§ £
¶ •• ß© ®® ™´ ™
≠ ¨¨ Æ
∞ Ø
± ØØ ≤≥ ≤≤ ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ª
æ ΩΩ ø
¿ øø ¡
√ ¬¬ ƒ∆ ≈≈ «» «
… ««  À    	 
           !  # %" &$ (  *) ,  .  0  2- 41 61 8> ;å =: ?> A/ B@ D5 F: G I: JE LK NM PH R T< US WQ YV ZX \S ^< a` d< e7 g: hf jc m≈ nH p rl sq uo wt xv zq |l H Å É~ ÑÇ ÜÄ àÖ âá ã< ç> è3 êé í  îì ñ≤ ôØ õö ù üò †û ¢ú §° ¶° ©® ´° ≠ö ∞ö ±ò ≥ µ¥ ∑≤ π∂ ∫∏ º
 æ ¿Ç √l ∆≈ »1 …« À  Ω ' ' )+ -+ ì9 :ï óï ΩC EC åó òO cO Që ìë :£ •£ ®i åi k[ ][ `ß Ø™ Ø™ ¨k l_ `b cª òª ΩÆ Øy {y ~} ~ä ¬ä ≈ƒ ≈  å  l– —— “  ŒŒ –“   œœ ÃÃ ÕÕ ¡–  –—  —Ω œœ Ω ÕÕ • ŒŒ •ø œœ ø ÃÃ  ÃÃ ¨ ŒŒ ¨ ÕÕ ” ¨
‘ —	’ -
’ ö
’ ú÷ ÷ ÷ •	◊ ◊ ]◊ {
◊ ì
◊ ®◊ Ø◊ ¡◊ ¬ÿ ÿ Ω	Ÿ 	Ÿ Ÿ :	Ÿ H	Ÿ M	Ÿ S	Ÿ q
Ÿ Ç
Ÿ ò
Ÿ û⁄ ⁄ 	⁄ )€ —	‹ 7› › øﬁ –	ﬂ 	ﬂ 5ﬂ <	ﬂ >	ﬂ K	ﬂ `	ﬂ ~
ﬂ å
ﬂ ≤
‡ ≈
· —"
_ZNSt8ios_base4InitC1Ev"
_ZNSt8ios_base4InitD1Ev"
__cxa_atexit"
main"
llvm.lifetime.start.p0i8"
scanf"
printf"
llvm.lifetime.end.p0i8"
_GLOBAL__sub_I__*x
-s
=
llvm_data_layout)
'
%e-m:e-i64:64-f80:128-n8:16:32:64-S128
2
llvm_target_triple

x86_64-unknown-linux-gnu2

poj104_label
5